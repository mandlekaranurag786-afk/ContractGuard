"""History page structure and security checks use the existing offline client."""

from html.parser import HTMLParser

from test_api_ui import PageParser
from test_api_ui import client as ui_client

client = ui_client


class StatusOptions(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.in_status = False
        self.values = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.in_status = attrs.get("id") == "history-status"
        elif tag == "option" and self.in_status:
            self.values.append(attrs.get("value"))

    def handle_endtag(self, tag):
        if tag == "select":
            self.in_status = False


def test_history_controls(client):
    response = client.get("/")
    assert response.status_code == 200
    page = PageParser(response.text)
    assert {
        "history-name", "history-status", "history-refresh", "history-note",
        "history-error", "history-table", "history-detail",
    } <= page.elements.keys()
    assert "Coming soon" not in response.text


def test_history_messages_and_requests(client):
    html = client.get("/").text
    for expected in (
        "/checks", "not_found", "URLSearchParams",
        "History stores decisions only, no quotes.",
        "History is turned off on this server.",
        "History stores full reports. Showing the newest 50 checks.",
    ):
        assert expected in html


def test_no_external_resources_or_unsafe_apis(client):
    html = client.get("/").text
    for forbidden in (
        "http://", "https://", "innerHTML", "outerHTML", "insertAdjacentHTML",
        "document.write", "eval(", "console.", "localStorage", "sessionStorage",
        "document.cookie",
    ):
        assert forbidden not in html


def test_history_status_options(client):
    assert StatusOptions(client.get("/").text).values == [
        "", "PASS", "FAIL", "NEEDS_REVIEW", "NO_APPLICABLE_RULES",
    ]
