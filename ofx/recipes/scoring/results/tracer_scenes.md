# The tracer's scenes: module vs effect (er_score tracerscenes)

12 scenes x 6 seeds, 112 x 112, one frame. module = traceStrands directly (frameHeight 1080, as test_strand_tracer); effect = Hair Locks through the host, Rebuild Detail only (vote, Snap, Temporal off; the region feathered 0.25 %, min piece scaled by the 112-line height), metrics on the alpha it ADDED; effect+snap = the same with Snap 1. rec needs the pieces (module only).

| scene | module rec | module false | effect false | effect+snap false | module distr | effect distr | effect+snap distr | module sad | effect sad | effect+snap sad |
|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.00 | 0.009 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.26 | 0.23 | 0.12 |
| pair | 1.00 | 0.002 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.22 | 0.26 | 0.16 |
| thin | 1.00 | 0.003 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | 0.43 | 0.41 | 0.20 |
| thick | 1.00 | 0.025 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.17 | 0.12 | 0.08 |
| light | 0.98 | 0.002 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.59 | 0.66 | 0.82 |
| crossing | 0.94 | 0.001 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.25 | 0.26 | 0.15 |
| glove | 1.00 | 0.015 | 0.000 | 0.008 | 0.000 | 0.000 | 0.002 | 0.26 | 0.20 | 0.08 |
| frame | 0.82 | 0.005 | 0.000 | 0.007 | 0.002 | 0.002 | 0.006 | 0.40 | 0.36 | 0.28 |
| brownframe | 0.86 | 0.005 | 0.000 | 0.007 | 0.002 | 0.002 | 0.006 | 0.37 | 0.33 | 0.25 |
| mole | 1.00 | 0.120 | 0.103 | 0.114 | 0.106 | 0.106 | 0.111 | 0.38 | 0.32 | 0.20 |
| texture | 1.00 | 0.003 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.25 | 0.23 | 0.13 |
| combo | 1.00 | 0.007 | 0.001 | 0.004 | 0.000 | 0.000 | 0.002 | 0.22 | 0.30 | 0.27 |
| **mean** | 0.97 | 0.016 | 0.009 | 0.014 | 0.009 | 0.009 | 0.011 | 0.32 | 0.31 | 0.23 |

Reference (docs/strand_tracing.md section 4, Python prototype, all rules): rec 0.97, false 0.01, distr 0.01, sad 0.30.
