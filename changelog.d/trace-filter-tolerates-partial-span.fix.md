`OTelTraceFilter` no longer raises when the active span does not implement
`get_span_context()`. Anything reporting itself as recording reaches the
filter — span shims, no-op spans, and test doubles implementing only the slice
of the OTel Span protocol their own caller needs — and a raising log filter
breaks logging for the entire process. A missing attribute now degrades to "no
trace id", honouring the "never raises" contract the class docstring already
stated.
