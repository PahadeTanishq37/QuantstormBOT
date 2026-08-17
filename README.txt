Divided Oracle -- open leaderboard client
===========================================

    pip install aiohttp
    python matchup.py --strategies /path/to/your/strategies

Your bot source code never leaves this machine -- only per-turn decisions
(bids, quotes, accept/counter, transform yes/no) go over the wire. Your
chosen strategies/*.py runs locally inside sandbox.py's process isolation,
the same as the real tournament.

There is no --name flag. Every strategies/*.py is required (RULEBOOK.md
SS12) to open with

    # Name: ...
    # College: ...
    # Roll Number: ...

and matchup.py reads your leaderboard identity straight off whichever of
your files has that filled in.

matchup.py prints a private dashboard link on connect -- open it and pick
AS MANY of your bots as you like. Up to 10 run at once (each
is a separate isolated process on YOUR machine); the rest queue and
activate automatically as slots free up. The server round-robins every
active strategy against everyone else's, automatically.

The public leaderboard shows one row per person: your single best-scoring
strategy. Your own private dashboard shows the full breakdown of all of
them.

matchup.py reconnects automatically (with backoff) if the connection drops
or the server restarts, and re-enters whatever you had selected -- no
manual restart needed.
