"""The `servecast` command line.

May import: every layer. It imports servecast.profile only inside the commands that measure, so every other
command runs without torch or vLLM.
"""
