use super::{CredentialRetrievalError, DockerCredential, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// 30-second budget for any single helper subprocess.
///
/// Public so a caller that needs a shorter budget can state its own against the
/// shipped one rather than re-spelling `30`.
pub const HELPER_TIMEOUT: Duration = Duration::from_secs(30);

/// 64 KiB cap on helper stdout.
const HELPER_OUTPUT_CAP: usize = 64 * 1024;

/// How often the wait loop polls the helper for exit.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Sentinel string the docker credential helper protocol emits on stdout when
/// no matching credential exists. Spec quirk: the message appears regardless
/// of exit code, and the protocol uses stdout (not stderr) for it.
const NOT_FOUND_SENTINEL: &str = "credentials not found in native keychain";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct HelperResponse {
    username: String,
    secret: String,
}

#[derive(serde::Serialize)]
struct StoreRequest<'a> {
    #[serde(rename = "ServerURL")]
    server_url: &'a str,
    #[serde(rename = "Username")]
    username: &'a str,
    #[serde(rename = "Secret")]
    secret: &'a str,
}

/// Spawn `docker-credential-{helper} {action}`, pipe `stdin`, return stdout bytes.
///
/// Generalized over the protocol action verb (`get`, `store`, `erase`, `list`).
/// Each public function in this module is a thin wrapper that calls `run_helper`
/// with action-specific payload and parses the result.
///
/// Behavior:
/// - Resolves the helper path via `resolve_helper_path` (PATH lookup + safety guard).
/// - Caps stdout at 64 KiB; oversized stdout returns `OutputTooLarge`.
/// - Enforces the [`HELPER_TIMEOUT`] subprocess budget; overrun kills the helper and returns `Timeout`.
/// - Sentinel string `"credentials not found in native keychain"` on stdout
///   maps to `NotFound` regardless of exit code.
/// - Non-zero exit without the sentinel returns `HelperFailure { stdout, stderr }`.
pub fn run_helper(helper: &str, action: &str, stdin_bytes: &[u8]) -> Result<Vec<u8>> {
    run_helper_with_timeout(helper, action, stdin_bytes, HELPER_TIMEOUT)
}

/// [`run_helper`] with the subprocess budget supplied by the caller.
///
/// Purely additive: `run_helper` is this function at [`HELPER_TIMEOUT`], and no
/// existing caller changes. It exists because the budget is the *only* thing a
/// test of the timeout path has to wait out — a helper that hangs is observed
/// identically at 30 s and at 200 ms, so a fixed constant made every such test,
/// this crate's own included, cost half a minute of real time to say something
/// about a deadline argument.
pub fn run_helper_with_timeout(
    helper: &str,
    action: &str,
    stdin_bytes: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>> {
    let path = resolve_helper_path(helper)?;
    let stdin_owned = stdin_bytes.to_vec();

    let mut child = Command::new(&path)
        .arg(action)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| CredentialRetrievalError::HelperCommunicationError)?;

    // Writing stdin in a dedicated thread avoids a deadlock when the helper
    // produces more stdout/stderr than the kernel pipe buffer can hold while
    // we are still blocked writing to stdin.
    let mut child_stdin = child
        .stdin
        .take()
        .ok_or(CredentialRetrievalError::HelperCommunicationError)?;
    let stdin_thread = thread::spawn(move || {
        let _ = child_stdin.write_all(&stdin_owned);
        // Closing stdin signals EOF to the helper.
        drop(child_stdin);
    });

    let child_stdout = child
        .stdout
        .take()
        .ok_or(CredentialRetrievalError::HelperCommunicationError)?;
    let child_stderr = child
        .stderr
        .take()
        .ok_or(CredentialRetrievalError::HelperCommunicationError)?;
    // Both pipes drain concurrently: reading one to EOF before the other lets a
    // helper block on the full second pipe and never exit (ocx-sh/ocx#586).
    let stdout_rx = drain_capped(child_stdout);
    let stderr_rx = drain_capped(child_stderr);

    let deadline = Instant::now() + timeout;
    let timed_out = || CredentialRetrievalError::Timeout {
        seconds: timeout.as_secs(),
    };
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(WAIT_POLL_INTERVAL),
            outcome => {
                // ponytail: kills the direct child only; kill the process group if helpers leak grandchildren.
                let _ = child.kill();
                let _ = child.wait();
                return Err(match outcome {
                    Ok(_) => timed_out(),
                    Err(_) => CredentialRetrievalError::HelperCommunicationError,
                });
            }
        }
    };
    let _ = stdin_thread.join();

    // A grandchild holding a pipe open outlives the helper; bound the drain by
    // the same budget instead of waiting on its EOF forever.
    let remaining = || deadline.saturating_duration_since(Instant::now());
    let stdout_buf = stdout_rx
        .recv_timeout(remaining())
        .map_err(|_| timed_out())?
        .map_err(|_| CredentialRetrievalError::HelperCommunicationError)?;
    // stderr only feeds the `HelperFailure` diagnostic; a read error there loses text, not the result.
    let stderr_buf = stderr_rx
        .recv_timeout(remaining())
        .map_err(|_| timed_out())?
        .unwrap_or_default();

    if stdout_buf.len() > HELPER_OUTPUT_CAP {
        return Err(CredentialRetrievalError::OutputTooLarge {
            cap_bytes: HELPER_OUTPUT_CAP,
        });
    }

    let stdout_text = String::from_utf8_lossy(&stdout_buf);
    let stderr_text = String::from_utf8_lossy(&stderr_buf).to_string();
    let trimmed_stdout = stdout_text.trim_end_matches(['\n', '\r']);
    if trimmed_stdout.lines().any(|line| line.trim() == NOT_FOUND_SENTINEL) {
        return Err(CredentialRetrievalError::NotFound);
    }

    if !status.success() {
        return Err(CredentialRetrievalError::HelperFailure {
            helper: format!("docker-credential-{helper}"),
            stdout: stdout_text.into_owned(),
            stderr: stderr_text,
        });
    }

    Ok(stdout_buf)
}

/// Read `pipe` to EOF on its own thread, keeping at most `HELPER_OUTPUT_CAP + 1`
/// bytes (one past the cap marks oversize) and discarding the rest, so the
/// helper never blocks on a full pipe.
fn drain_capped(mut pipe: impl Read + Send + 'static) -> mpsc::Receiver<io::Result<Vec<u8>>> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut buf = Vec::new();
        let result = (&mut pipe)
            .take(HELPER_OUTPUT_CAP as u64 + 1)
            .read_to_end(&mut buf)
            .and_then(|_| io::copy(&mut pipe, &mut io::sink()))
            .map(|_| buf);
        // The receiver is gone only after a timeout already returned.
        let _ = tx.send(result);
    });
    rx
}

fn response_from_helper(address: &str, helper: &str, timeout: Duration) -> Result<HelperResponse> {
    let output = run_helper_with_timeout(helper, "get", address.as_bytes(), timeout)?;
    serde_json::from_slice(&output).map_err(|_| CredentialRetrievalError::MalformedHelperResponse)
}

pub fn credential_from_helper(address: &str, helper: &str) -> Result<DockerCredential> {
    credential_from_helper_with_timeout(address, helper, HELPER_TIMEOUT)
}

/// [`credential_from_helper`] with the subprocess budget supplied by the caller.
pub fn credential_from_helper_with_timeout(
    address: &str,
    helper: &str,
    timeout: Duration,
) -> Result<DockerCredential> {
    let response = response_from_helper(address, helper, timeout)?;

    if response.username == "<token>" {
        Ok(DockerCredential::IdentityToken(response.secret))
    } else {
        Ok(DockerCredential::UsernamePassword(
            response.username,
            response.secret,
        ))
    }
}

/// Persist `cred` for `server` to the named helper's backing store.
///
/// Serializes `{ServerURL, Username, Secret}` as PascalCase JSON on stdin.
/// `DockerCredential::IdentityToken(tok)` is encoded with `Username = "<token>"`
/// and `Secret = tok` to match the helper protocol's identity-token wire form.
pub fn store_credential(
    server: &str,
    helper: &str,
    cred: &DockerCredential,
) -> Result<()> {
    store_credential_with_timeout(server, helper, cred, HELPER_TIMEOUT)
}

/// [`store_credential`] with the subprocess budget supplied by the caller.
pub fn store_credential_with_timeout(
    server: &str,
    helper: &str,
    cred: &DockerCredential,
    timeout: Duration,
) -> Result<()> {
    let (username, secret) = match cred {
        DockerCredential::IdentityToken(token) => ("<token>", token.as_str()),
        DockerCredential::UsernamePassword(user, pwd) => (user.as_str(), pwd.as_str()),
    };
    let payload = StoreRequest {
        server_url: server,
        username,
        secret,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|_| CredentialRetrievalError::HelperCommunicationError)?;
    run_helper_with_timeout(helper, "store", &bytes, timeout)?;
    Ok(())
}

/// Remove a credential for `server` from the named helper.
///
/// Writes the raw server URL bytes to stdin (protocol quirk shared with `get`).
/// A `NotFound` sentinel from the helper is treated as already-erased and
/// surfaces as `Ok(())` so callers can use `erase` idempotently.
pub fn erase_credential(server: &str, helper: &str) -> Result<()> {
    erase_credential_with_timeout(server, helper, HELPER_TIMEOUT)
}

/// [`erase_credential`] with the subprocess budget supplied by the caller.
pub fn erase_credential_with_timeout(server: &str, helper: &str, timeout: Duration) -> Result<()> {
    match run_helper_with_timeout(helper, "erase", server.as_bytes(), timeout) {
        Ok(_) => Ok(()),
        Err(CredentialRetrievalError::NotFound) => Ok(()),
        Err(err) => Err(err),
    }
}

/// List all credentials known to the named helper.
///
/// Returns the `{serverURL: username}` map the helper emits on stdout.
pub fn list_credentials(helper: &str) -> Result<HashMap<String, String>> {
    let output = run_helper(helper, "list", b"")?;
    serde_json::from_slice(&output).map_err(CredentialRetrievalError::InvalidJson)
}

/// Probe PATH for the platform-default credential helper.
///
/// Returns the suffix used as `credsStore` in `~/.docker/config.json`:
/// macOS → `osxkeychain`; Windows → `wincred`; Linux → first of `pass`,
/// `secretservice`. `None` when no candidate is on PATH or resolves under
/// an unsafe location.
pub fn detect_default_helper() -> Option<String> {
    #[cfg(target_os = "macos")]
    let candidates: &[&str] = &["osxkeychain"];
    #[cfg(target_os = "windows")]
    let candidates: &[&str] = &["wincred"];
    #[cfg(target_os = "linux")]
    let candidates: &[&str] = &["pass", "secretservice"];
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    let candidates: &[&str] = &[];

    for &candidate in candidates {
        if resolve_helper_path(candidate).is_ok() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Resolve a helper suffix to its absolute binary path, rejecting unsafe locations.
///
/// Algorithm:
/// 1. `which::which("docker-credential-{suffix}")`. Missing → `NotOnPath`.
/// 2. Canonicalize the resolved path.
/// 3. Reject if any ancestor directory of the canonical path is world-writable
///    (others-write bit set on Unix). The acceptance tests rely on this guard
///    to reject helpers planted in `/tmp/<random>` (intentionally 0o777) while
///    still resolving helpers under `tempfile::tempdir()` (0o700) used by
///    standard unit-test fixtures.
pub(crate) fn resolve_helper_path(suffix: &str) -> Result<PathBuf> {
    let helper_name = format!("docker-credential-{suffix}");
    let resolved = which::which(&helper_name).map_err(|_| CredentialRetrievalError::NotOnPath {
        name: suffix.to_string(),
    })?;
    let canonical = resolved.canonicalize().unwrap_or(resolved);

    if path_has_world_writable_ancestor(&canonical) {
        return Err(CredentialRetrievalError::UnsafePath {
            name: suffix.to_string(),
            path: canonical,
        });
    }

    Ok(canonical)
}

/// Returns true if any ancestor directory of `path` is unsafely writable.
///
/// A directory counts as unsafe when its `others` write bit is set AND the
/// sticky bit (S_ISVTX) is not. The sticky bit on `/tmp` (mode 0o1777) keeps
/// it usable for legitimate helper paths under per-user subtrees; a directory
/// with mode 0o777 and no sticky bit can be co-opted by any local user to
/// plant a tampered binary, which the PATH lookup would then resolve.
///
/// On non-Unix platforms always returns false (Windows ACL model differs).
#[cfg(unix)]
fn path_has_world_writable_ancestor(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    let mut current: Option<&Path> = path.parent();
    while let Some(dir) = current {
        if let Ok(meta) = std::fs::metadata(dir) {
            let mode = meta.permissions().mode();
            let world_writable = mode & 0o002 != 0;
            let sticky = mode & 0o1000 != 0;
            if world_writable && !sticky {
                return true;
            }
        }
        let next = dir.parent();
        if next == Some(dir) {
            break;
        }
        current = next;
    }
    false
}

#[cfg(not(unix))]
fn path_has_world_writable_ancestor(_path: &Path) -> bool {
    false
}

// ─────────────────────────── tests ───────────────────────────
//
// Tests pin the public protocol surface for the write API extension:
//   * `run_helper(name, action, stdin) -> Result<Vec<u8>>`
//   * `store_credential` / `erase_credential` / `list_credentials` wrappers
//   * `detect_default_helper` / `resolve_helper_path` security guards
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt as _;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serialises every test that mutates `PATH` — multiple `PATH` rewrites in
    /// parallel race because the env is process-global.
    static PATH_LOCK: Mutex<()> = Mutex::new(());

    /// Builds a mock `docker-credential-test` helper script into a fresh tempdir.
    /// Returns `(tempdir guard, absolute binary path)`. The dir holds the binary
    /// and a sidecar file for stdin capture.
    #[cfg(unix)]
    fn make_mock_helper(script_body: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let bin = dir.path().join("docker-credential-test");
        let body = format!("#!/bin/sh\n{script_body}\n");
        fs::write(&bin, body).expect("write helper script");
        let mut perms = fs::metadata(&bin).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).expect("chmod 0755");
        (dir, bin)
    }

    /// Prepends `dir` to the current `PATH` for the duration of the closure.
    /// PATH is saved + restored. The caller must hold `PATH_LOCK`.
    #[cfg(unix)]
    fn with_path_prepended<F: FnOnce()>(dir: &std::path::Path, body: F) {
        let original = env::var_os("PATH");
        let mut new = std::ffi::OsString::from(dir);
        if let Some(orig) = &original {
            new.push(":");
            new.push(orig);
        }
        // SAFETY: serialised by `PATH_LOCK`; restored in scope below.
        unsafe { env::set_var("PATH", &new) };
        body();
        // SAFETY: see above.
        unsafe {
            match original {
                Some(p) => env::set_var("PATH", p),
                None => env::remove_var("PATH"),
            }
        }
    }

    // ─── run_helper ───

    #[test]
    #[cfg(unix)]
    fn run_helper_get_succeeds_with_valid_response() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper(
            r#"echo '{"Username":"u","Secret":"p"}'"#,
        );
        with_path_prepended(dir.path(), || {
            let out = run_helper("test", "get", b"ghcr.io").expect("run_helper get");
            let s = String::from_utf8(out).expect("utf8");
            assert!(s.contains("\"Username\":\"u\""), "stdout was: {s}");
            assert!(s.contains("\"Secret\":\"p\""), "stdout was: {s}");
        });
    }

    #[test]
    #[cfg(unix)]
    fn run_helper_store_writes_pascalcase_json_to_stdin() {
        let _g = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("create tempdir");
        let sidecar = dir.path().join("stdin.json");
        let bin = dir.path().join("docker-credential-test");
        let script = format!(
            "#!/bin/sh\ncat > {sidecar}\n",
            sidecar = sidecar.display(),
        );
        fs::write(&bin, script).expect("write helper");
        let mut perms = fs::metadata(&bin).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).expect("chmod");

        let cred = DockerCredential::UsernamePassword("u".into(), "p".into());
        with_path_prepended(dir.path(), || {
            store_credential("ghcr.io", "test", &cred).expect("store");
        });

        let captured = fs::read_to_string(&sidecar).expect("read sidecar");
        let value: serde_json::Value = serde_json::from_str(&captured).expect("json");
        assert_eq!(value.get("ServerURL").and_then(|v| v.as_str()), Some("ghcr.io"));
        assert_eq!(value.get("Username").and_then(|v| v.as_str()), Some("u"));
        assert_eq!(value.get("Secret").and_then(|v| v.as_str()), Some("p"));
    }

    #[test]
    #[cfg(unix)]
    fn run_helper_erase_writes_raw_url_to_stdin() {
        let _g = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("create tempdir");
        let sidecar = dir.path().join("stdin.txt");
        let bin = dir.path().join("docker-credential-test");
        let script = format!(
            "#!/bin/sh\ncat > {sidecar}\n",
            sidecar = sidecar.display(),
        );
        fs::write(&bin, script).expect("write helper");
        let mut perms = fs::metadata(&bin).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).expect("chmod");

        with_path_prepended(dir.path(), || {
            erase_credential("ghcr.io", "test").expect("erase");
        });

        let captured = fs::read(&sidecar).expect("read sidecar");
        assert_eq!(captured, b"ghcr.io", "erase must pass raw URL bytes (no JSON)");
    }

    #[test]
    #[cfg(unix)]
    fn run_helper_sentinel_string_maps_to_not_found() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper(
            "echo 'credentials not found in native keychain'\nexit 1",
        );
        with_path_prepended(dir.path(), || {
            let err = run_helper("test", "get", b"ghcr.io").expect_err("should error");
            assert!(
                matches!(err, CredentialRetrievalError::NotFound),
                "expected NotFound, got: {err:?}",
            );
        });
    }

    /// The shipped budget is 30 s, asserted as a constant rather than waited out.
    ///
    /// Split from the behavioural row below deliberately. The two claims are
    /// independent — *what* the budget is, and *that* it fires — and the old
    /// single test could only make the first by paying the second in real time:
    /// it slept a helper for 60 s and asserted `Timeout { seconds: 30 }`, which
    /// cost ~30 s of every `cargo test` run on this crate.
    #[test]
    fn helper_timeout_is_thirty_seconds() {
        assert_eq!(HELPER_TIMEOUT, Duration::from_secs(30));
    }

    /// The budget fires, and it fires at the value it was handed.
    ///
    /// Driven through `run_helper_with_timeout` at 200 ms: the code path is
    /// a deadline check, which cannot tell one `Duration` from
    /// another, so a short budget observes exactly what a long one would. The
    /// helper still sleeps far past it, which is what makes the pass mean the
    /// timeout fired rather than the child exiting on its own.
    #[test]
    #[cfg(unix)]
    fn run_helper_timeout_fires_at_the_budget_it_is_given() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper("sleep 60");
        with_path_prepended(dir.path(), || {
            let start = std::time::Instant::now();
            let err = run_helper_with_timeout(
                "test",
                "get",
                b"ghcr.io",
                Duration::from_millis(200),
            )
            .expect_err("must time out");
            let elapsed = start.elapsed();
            assert!(
                matches!(err, CredentialRetrievalError::Timeout { seconds: 0 }),
                "expected Timeout {{ seconds: 0 }} (200ms truncates), got: {err:?}",
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "timeout fired late ({:?})",
                elapsed,
            );
        });
    }

    /// A helper that fills its stderr pipe before closing stdout must not deadlock.
    #[test]
    #[cfg(unix)]
    fn run_helper_drains_large_stderr() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper(
            "dd if=/dev/zero bs=1024 count=128 2>/dev/null | tr '\\0' 'e' >&2\n\
             echo '{\"Username\":\"u\",\"Secret\":\"p\"}'",
        );
        with_path_prepended(dir.path(), || {
            run_helper_with_timeout("test", "get", b"ghcr.io", Duration::from_secs(5))
                .expect("128 KiB of stderr must not block the helper");
        });
    }

    /// A helper that overruns its budget is killed, not left running.
    #[test]
    #[cfg(unix)]
    fn run_helper_timeout_kills_child() {
        let _g = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("create tempdir");
        let pid_file = dir.path().join("pid");
        let bin = dir.path().join("docker-credential-test");
        fs::write(&bin, format!("#!/bin/sh\necho $$ > {}\nexec sleep 60\n", pid_file.display()))
            .expect("write helper");
        let mut perms = fs::metadata(&bin).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).expect("chmod");

        with_path_prepended(dir.path(), || {
            let err = run_helper_with_timeout("test", "get", b"ghcr.io", Duration::from_millis(500))
                .expect_err("must time out");
            assert!(matches!(err, CredentialRetrievalError::Timeout { .. }), "got: {:?}", err);
        });
        let pid = fs::read_to_string(&pid_file).expect("helper wrote its pid");
        let alive = std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .status()
            .expect("run kill -0")
            .success();
        assert!(!alive, "timed-out helper {} is still running", pid.trim());
    }

    #[test]
    #[cfg(unix)]
    fn run_helper_output_cap_at_64kib() {
        let _g = PATH_LOCK.lock().unwrap();
        // Emit 128 KiB to stdout (printf 'a' 131072 times via head). We use
        // `dd` so the script stays portable.
        let (dir, _bin) = make_mock_helper(
            "dd if=/dev/zero bs=1024 count=128 2>/dev/null | tr '\\0' 'a'",
        );
        with_path_prepended(dir.path(), || {
            let err = run_helper("test", "get", b"ghcr.io").expect_err("must reject oversize");
            assert!(
                matches!(err, CredentialRetrievalError::OutputTooLarge { cap_bytes: 65536 }),
                "expected OutputTooLarge {{ cap_bytes: 65536 }}, got: {err:?}",
            );
        });
    }

    #[test]
    #[cfg(unix)]
    fn run_helper_non_zero_exit_with_empty_stdout() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper("echo error-msg >&2\nexit 1");
        with_path_prepended(dir.path(), || {
            let err = run_helper("test", "get", b"ghcr.io").expect_err("must fail");
            match err {
                CredentialRetrievalError::HelperFailure { stdout, stderr, .. } => {
                    assert!(stdout.is_empty(), "stdout: {stdout:?}");
                    assert!(stderr.contains("error-msg"), "stderr: {stderr:?}");
                }
                other => panic!("expected HelperFailure, got: {:?}", other),
            }
        });
    }

    // ─── resolve_helper_path ───

    #[test]
    #[cfg(unix)]
    fn resolve_helper_path_finds_helper_on_path() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, bin) = make_mock_helper("echo ok");
        with_path_prepended(dir.path(), || {
            // Suffix "test" → looks up `docker-credential-test`.
            let resolved = resolve_helper_path("test").expect("should resolve");
            assert_eq!(
                resolved.canonicalize().expect("canonicalize"),
                bin.canonicalize().expect("canonicalize"),
            );
        });
    }

    #[test]
    #[cfg(unix)]
    fn resolve_helper_path_rejects_world_writable_dir() {
        let _g = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir_in("/tmp").expect("tempdir in /tmp");
        let mut perms = fs::metadata(dir.path()).expect("meta").permissions();
        perms.set_mode(0o777);
        fs::set_permissions(dir.path(), perms).expect("chmod 0777");
        let bin = dir.path().join("docker-credential-test");
        fs::File::create(&bin)
            .expect("create bin")
            .write_all(b"#!/bin/sh\necho ok\n")
            .expect("write bin");
        let mut bp = fs::metadata(&bin).expect("meta").permissions();
        bp.set_mode(0o755);
        fs::set_permissions(&bin, bp).expect("chmod");

        with_path_prepended(dir.path(), || {
            let err = resolve_helper_path("test").expect_err("must reject unsafe path");
            assert!(
                matches!(err, CredentialRetrievalError::UnsafePath { .. }),
                "expected UnsafePath, got: {err:?}",
            );
        });
    }

    #[test]
    #[cfg(unix)]
    fn resolve_helper_path_not_on_path() {
        let _g = PATH_LOCK.lock().unwrap();
        // Point PATH at an empty tempdir — the helper cannot resolve.
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: serialised by PATH_LOCK
        let original = env::var_os("PATH");
        unsafe { env::set_var("PATH", dir.path()) };
        let result = resolve_helper_path("nonexistent-binary-name");
        // SAFETY: restore
        unsafe {
            match original {
                Some(p) => env::set_var("PATH", p),
                None => env::remove_var("PATH"),
            }
        }
        let err = result.expect_err("must fail");
        assert!(
            matches!(err, CredentialRetrievalError::NotOnPath { .. }),
            "expected NotOnPath, got: {err:?}",
        );
    }

    // ─── detect_default_helper ───

    #[test]
    fn detect_default_helper_returns_platform_appropriate() {
        // Per-platform expectation:
        //   - macOS expects Some("osxkeychain") if installed, else None
        //   - Linux expects "pass" | "secretservice" | None
        //   - Windows expects "wincred" | None
        let detected = detect_default_helper();
        #[cfg(target_os = "macos")]
        {
            assert!(
                detected.is_none() || detected.as_deref() == Some("osxkeychain"),
                "macOS detected helper: {:?}",
                detected,
            );
        }
        #[cfg(target_os = "linux")]
        {
            assert!(
                detected.is_none()
                    || detected.as_deref() == Some("pass")
                    || detected.as_deref() == Some("secretservice"),
                "linux detected helper: {:?}",
                detected,
            );
        }
        #[cfg(target_os = "windows")]
        {
            assert!(
                detected.is_none() || detected.as_deref() == Some("wincred"),
                "windows detected helper: {:?}",
                detected,
            );
        }
    }

    // ─── store_credential identity-token variant ───

    #[test]
    #[cfg(unix)]
    fn store_credential_username_token_encodes_identity_token() {
        let _g = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("create tempdir");
        let sidecar = dir.path().join("stdin.json");
        let bin = dir.path().join("docker-credential-test");
        let script = format!(
            "#!/bin/sh\ncat > {sidecar}\n",
            sidecar = sidecar.display(),
        );
        fs::write(&bin, script).expect("write helper");
        let mut perms = fs::metadata(&bin).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).expect("chmod");

        let cred = DockerCredential::IdentityToken("tok".into());
        with_path_prepended(dir.path(), || {
            store_credential("ghcr.io", "test", &cred).expect("store identity token");
        });
        let captured = fs::read_to_string(&sidecar).expect("read sidecar");
        let value: serde_json::Value = serde_json::from_str(&captured).expect("json");
        assert_eq!(value.get("Username").and_then(|v| v.as_str()), Some("<token>"));
        assert_eq!(value.get("Secret").and_then(|v| v.as_str()), Some("tok"));
    }

    // ─── list_credentials ───

    #[test]
    #[cfg(unix)]
    fn list_credentials_parses_json_map() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper(r#"echo '{"a":"u1","b":"u2"}'"#);
        with_path_prepended(dir.path(), || {
            let map = list_credentials("test").expect("list");
            assert_eq!(map.get("a").map(String::as_str), Some("u1"));
            assert_eq!(map.get("b").map(String::as_str), Some("u2"));
        });
    }

    #[test]
    #[cfg(unix)]
    fn list_credentials_invalid_json_returns_error() {
        let _g = PATH_LOCK.lock().unwrap();
        let (dir, _bin) = make_mock_helper("echo not-json");
        with_path_prepended(dir.path(), || {
            let err = list_credentials("test").expect_err("must fail");
            assert!(
                matches!(err, CredentialRetrievalError::InvalidJson(_)),
                "expected InvalidJson, got: {err:?}",
            );
            // Confirm Error::source() chain is populated for diagnostics.
            let std_err: &dyn std::error::Error = &err;
            assert!(std_err.source().is_some(), "InvalidJson should expose source");
        });
    }
}
