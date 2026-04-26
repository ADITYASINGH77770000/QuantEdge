class _ColumnStub:
    def metric(self, *args, **kwargs):
        return None

class _SidebarStub:
    def __getattr__(self, item):
        def _noop(*args, **kwargs):
            return None
        return _noop

class StreamlitStub:
    def markdown(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None

    def set_page_config(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def columns(self, n=1):
        return [_ColumnStub() for _ in range(n)]

    def dataframe(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def sidebar(self):
        return _SidebarStub()

# singleton instance exported as `st`
st = StreamlitStub()
