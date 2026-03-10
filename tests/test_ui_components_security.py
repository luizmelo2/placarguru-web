import ui_components as ui


def test_render_chip_escapes_html_content():
    out = ui.render_chip('<img src=x onerror=alert(1)>', tone='ghost')
    assert '<img' not in out
    assert '&lt;img src=x onerror=alert(1)&gt;' in out


def test_render_app_header_escapes_live_messages():
    out = ui.render_app_header(live_messages=['ok', '<script>alert(1)</script>'])
    assert '<script>' not in out
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in out
