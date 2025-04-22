#!/usr/bin/env python3
"""
PyPSA MCP Server

Main entry point for running the MCP server for PyPSA energy model creation and analysis.
"""

import sys
from pypsamcp.core import mcp

def main():
    """Main entry point for the PyPSA MCP server."""
    
    try:
        print("Starting PyPSA MCP Server...")
        
        mcp.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
