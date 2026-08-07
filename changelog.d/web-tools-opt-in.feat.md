Claude SDK agents can now be granted `WebFetch` / `WebSearch` while otherwise
restricted, via the new `web_tools=True` argument. They were previously denied
along with the filesystem and shell built-ins, so a restricted research agent had
no way to look anything up — and a refused tool call is indistinguishable from an
empty result, so it reported "sources fetched, all empty" instead of "I cannot
search". Reading the web mutates nothing local, so it is separable from the
sandbox the denylist exists to enforce. Default stays off.
