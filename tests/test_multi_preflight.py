"""
Regression tests for `shot-scraper multi` pre-flight validation, per-shot
isolation, atomic output publishing and the run manifest.

Covers:
- Bad configuration causes zero side effects (no sh/python/server run)
- Duplicate output filenames are rejected before anything runs
- Timeouts are recorded and later shots still run
- Partial failures are recorded and the run continues
- Authentication state is shared across shots while page state is isolated
- Re-runs behave (no-clobber skips, failed shots are retried, stable manifest)
"""

import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

from shot_scraper.cli import cli


@pytest.fixture
def recording_server():
    """HTTP server that records the path and Cookie header of each request."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(
                {"path": self.path, "cookie": self.headers.get("Cookie") or ""}
            )
            body = b"<html><body><h1>Hello</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.parametrize(
    "bad_yaml",
    (
        # Unknown key in a later shot
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  bogus_key: 1\n",
        # output without url
        "- sh: touch side-effect.txt\n" "- output: out.png\n",
        # js_file referencing a missing file
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  js_file: missing.js\n",
        # javascript and js_file used together
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  javascript: '1'\n"
        "  js_file: whatever.js\n",
        # format: png combined with quality
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  output: out.png\n"
        "  format: png\n"
        "  quality: 80\n",
        # wrong type for width
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  width: wide\n",
        # wrong type for sh
        "- sh: touch side-effect.txt\n" "- sh: 123\n",
        # empty shot
        "- sh: touch side-effect.txt\n" "- {}\n",
        # shot that is not a mapping
        "- sh: touch side-effect.txt\n" "- just a string\n",
    ),
)
def test_bad_config_zero_side_effects(bad_yaml):
    # A config error anywhere must abort before any sh/python/server runs
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(bad_yaml)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code != 0
        assert not pathlib.Path("side-effect.txt").exists()
        assert not pathlib.Path("manifest.json").exists()
        assert list(pathlib.Path(".").glob("*.png")) == []


def test_duplicate_outputs_rejected_before_side_effects():
    runner = CliRunner()
    yaml_text = (
        "- sh: touch side-effect.txt\n"
        "- url: http://example.com/\n"
        "  output: same.png\n"
        "- url: http://example.org/\n"
        "  output: same.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code != 0
        assert "Duplicate output" in result.output
        assert "same.png" in result.output
        assert not pathlib.Path("side-effect.txt").exists()
        assert not pathlib.Path("manifest.json").exists()
        assert not pathlib.Path("same.png").exists()


def test_duplicate_derived_outputs_rejected():
    # Two shots of the same URL with no output: derive the same filename
    runner = CliRunner()
    yaml_text = "- url: http://example.com/\n- url: http://example.com/\n"
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code != 0
        assert "Duplicate output" in result.output
        assert not pathlib.Path("manifest.json").exists()


def test_output_conflicting_with_manifest_rejected():
    runner = CliRunner()
    yaml_text = "- url: http://example.com/\n  output: manifest.json\n"
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code != 0
        assert "manifest" in result.output
        assert not pathlib.Path("manifest.json").exists()


def test_timeout_recorded_and_run_continues(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        '  wait_for: "false"\n'
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml", "--timeout", "300"])
        assert result.exit_code == 0, result.output
        # The timed-out shot produced no output file, the later shot did
        assert not pathlib.Path("one.png").exists()
        assert pathlib.Path("two.png").exists()
        # No temporary files were left behind
        assert list(pathlib.Path(".").glob("*.png.*")) == []
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["failed", "success"]
        assert "Timeout" in manifest["shots"][0]["reason"]
        assert manifest["shots"][0]["url"] == f"{base_url}/one"
        assert manifest["shots"][0]["output"] == "one.png"
        assert manifest["summary"] == {"failed": 1, "success": 1}


def test_partial_failure_recorded_and_run_continues(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        # Nothing is listening on port 1
        "- url: http://127.0.0.1:1/nope\n"
        "  output: two.png\n"
        f"- url: {base_url}/three\n"
        "  output: three.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code == 0, result.output
        assert pathlib.Path("one.png").exists()
        assert not pathlib.Path("two.png").exists()
        assert pathlib.Path("three.png").exists()
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == [
            "success",
            "failed",
            "success",
        ]
        assert manifest["shots"][1]["reason"]


def test_fail_option_aborts_run(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        "- url: http://127.0.0.1:1/nope\n"
        "  output: two.png\n"
        f"- url: {base_url}/three\n"
        "  output: three.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml", "--fail"])
        assert result.exit_code != 0
        assert pathlib.Path("one.png").exists()
        # The run aborted, so the third shot never happened
        assert not pathlib.Path("three.png").exists()
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == [
            "success",
            "failed",
            "pending",
        ]


def test_shared_auth_state_across_shots(recording_server):
    base_url, requests = recording_server
    runner = CliRunner()
    auth = {
        "cookies": [
            {
                "name": "token",
                "value": "secret",
                "domain": "127.0.0.1",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": False,
                "sameSite": "Lax",
            }
        ],
        "origins": [],
    }
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("auth.json").write_text(json.dumps(auth))
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml", "-a", "auth.json"])
        assert result.exit_code == 0, result.output
    assert [r["path"] for r in requests] == ["/one", "/two"]
    assert all("token=secret" in r["cookie"] for r in requests)


def test_context_shared_but_pages_isolated(recording_server):
    # Cookies set by one shot's JavaScript are visible to later shots
    # (shared browser context), but JavaScript globals do not leak
    # between shots (isolated pages)
    base_url, requests = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        "  javascript: |\n"
        "    document.cookie = 'shared=1';\n"
        "    window.__leaked = true;\n"
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
        "  javascript: |\n"
        "    if (window.__leaked) { throw new Error('page state leaked'); }\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code == 0, result.output
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["success", "success"]
    assert "shared=1" in requests[1]["cookie"]


def test_rerun_with_noclobber_and_stable_manifest(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code == 0, result.output
        first_manifest = pathlib.Path("manifest.json").read_text()
        one_png = pathlib.Path("one.png").read_bytes()
        two_png = pathlib.Path("two.png").read_bytes()

        # Re-running produces an identical manifest
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code == 0, result.output
        assert pathlib.Path("manifest.json").read_text() == first_manifest

        # Re-running with --no-clobber skips existing outputs and says so
        result = runner.invoke(cli, ["multi", "shots.yml", "-n"])
        assert result.exit_code == 0, result.output
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["skipped", "skipped"]
        assert all("--no-clobber" in s["reason"] for s in manifest["shots"])
        # Outputs were not modified
        assert pathlib.Path("one.png").read_bytes() == one_png
        assert pathlib.Path("two.png").read_bytes() == two_png


def test_rerun_retries_failed_shots(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        "- url: http://127.0.0.1:1/nope\n"
        "  output: one.png\n"
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml"])
        assert result.exit_code == 0, result.output
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["failed", "success"]
        two_png = pathlib.Path("two.png").read_bytes()

        # With -n the failed shot is retried, the successful one is skipped
        result = runner.invoke(cli, ["multi", "shots.yml", "-n"])
        assert result.exit_code == 0, result.output
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["failed", "skipped"]
        assert pathlib.Path("two.png").read_bytes() == two_png


def test_output_filter_marks_unselected_as_skipped(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = (
        f"- url: {base_url}/one\n"
        "  output: one.png\n"
        f"- url: {base_url}/two\n"
        "  output: two.png\n"
    )
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml", "-o", "one.png"])
        assert result.exit_code == 0, result.output
        assert pathlib.Path("one.png").exists()
        assert not pathlib.Path("two.png").exists()
        manifest = json.loads(pathlib.Path("manifest.json").read_text())
        assert [s["status"] for s in manifest["shots"]] == ["success", "skipped"]
        assert manifest["shots"][1]["reason"] == "not selected by --output"


def test_manifest_option(recording_server):
    base_url, _ = recording_server
    runner = CliRunner()
    yaml_text = f"- url: {base_url}/one\n  output: one.png\n"
    with runner.isolated_filesystem():
        pathlib.Path("shots.yml").write_text(yaml_text)
        result = runner.invoke(cli, ["multi", "shots.yml", "--manifest", "custom.json"])
        assert result.exit_code == 0, result.output
        assert not pathlib.Path("manifest.json").exists()
        manifest = json.loads(pathlib.Path("custom.json").read_text())
        assert manifest["shots"][0]["status"] == "success"


def test_skip_option_marks_http_errors_as_skipped():
    runner = CliRunner()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == "/missing":
                self.send_response(404)
                self.end_headers()
            else:
                body = b"<h1>ok</h1>"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        yaml_text = (
            f"- url: {base_url}/missing\n"
            "  output: missing.png\n"
            f"- url: {base_url}/ok\n"
            "  output: ok.png\n"
        )
        with runner.isolated_filesystem():
            pathlib.Path("shots.yml").write_text(yaml_text)
            result = runner.invoke(cli, ["multi", "shots.yml", "--skip"])
            assert result.exit_code == 0, result.output
            assert not pathlib.Path("missing.png").exists()
            assert pathlib.Path("ok.png").exists()
            manifest = json.loads(pathlib.Path("manifest.json").read_text())
            assert [s["status"] for s in manifest["shots"]] == [
                "skipped",
                "success",
            ]
    finally:
        server.shutdown()
        thread.join()
