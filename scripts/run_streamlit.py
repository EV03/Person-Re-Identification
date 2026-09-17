from __future__ import annotations

import asyncio
import sys


def _ignore_expected_windows_connection_resets(loop: asyncio.AbstractEventLoop) -> None:
    """Hide the harmless Proactor callback traceback produced by closed browser sockets."""

    def handle_exception(current_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        exception = context.get("exception")
        if isinstance(exception, ConnectionResetError) and getattr(exception, "winerror", None) == 10054:
            return
        current_loop.default_exception_handler(context)

    loop.set_exception_handler(handle_exception)


if sys.platform == "win32":
    base_policy = asyncio.WindowsProactorEventLoopPolicy

    class QuietWindowsProactorEventLoopPolicy(base_policy):
        def new_event_loop(self) -> asyncio.AbstractEventLoop:
            loop = super().new_event_loop()
            _ignore_expected_windows_connection_resets(loop)
            return loop

    asyncio.set_event_loop_policy(QuietWindowsProactorEventLoopPolicy())


from streamlit.web.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
