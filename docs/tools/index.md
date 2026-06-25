# robotsix_llmio tools

Built-in example tools for out-of-the-box agent functionality.

## Exports

- `get_builtin_tools` — returns a list of the four built-in callables
  (`get_time`, `echo`, `calculator`, `roll_dice`) ready to pass to
  `build_agent(tools=...)`

### Built-in tools

- `get_time` — returns the current time as an ISO-8601 string
- `echo` — identity tool: returns its input unchanged
- `calculator` — evaluates a safe arithmetic expression via AST whitelist
- `roll_dice` — returns a random integer from 1 to `sides` (default 6)
