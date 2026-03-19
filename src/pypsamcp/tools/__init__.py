"""
PyPSA MCP Tool modules.

Importing this package triggers registration of all @mcp.tool() decorated
functions from each submodule.
"""

from pypsamcp.tools import management  # noqa: F401
from pypsamcp.tools import discovery  # noqa: F401
from pypsamcp.tools import components  # noqa: F401
from pypsamcp.tools import convenience  # noqa: F401
from pypsamcp.tools import time_config  # noqa: F401
from pypsamcp.tools import simulation  # noqa: F401
from pypsamcp.tools import statistics  # noqa: F401
from pypsamcp.tools import clustering  # noqa: F401
from pypsamcp.tools import io  # noqa: F401
from pypsamcp.tools import deprecated  # noqa: F401
