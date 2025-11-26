# "ESI-Proxy"

For when you've been making a project with ESI-data and they add rate limiting,
or you want to start working offline more, and you just can't be bothered to
update your original code, so you just vibe-code a proxy that handles the
error and rate limits, and caches resources so you don't make silly requests
every time you do python my_fancy_app.py just to test if the print() debugs now work.

Just uuuh update your endpoint URIs in code to point to the proxy and carry on.

-----------------------------

"Coercing claude to make a drop-in replacement since 2025"

Problem is, I write some code, Claude looks over it, makes some changes, and it becomes
maddeningly unreadable and now I have to slowly clean it up.



Just have to remember this for myself;
rsync -av --filter=':- .gitignore' ./ dest/
