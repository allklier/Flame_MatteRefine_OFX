# Hair Locks timing (er_score timing)

CPU: 12th Gen Intel(R) Core(TM) i9-12900KS; cores allowed to this process: 2 (affinity); std::thread::hardware_concurrency() = 24 (the cores' worker pools size by this, so their threads time-share the allowed cores).

Every classical step on (Rebuild Detail + Temporal Vote, Snap, Temporal +-1), Edge Model and Jump Repair off; 5-frame clip (a glove distractor, base jitter 0.75 px), centre frame rendered 3 times; wall ms through the host (input copy + render + output), then the plugin's per-step ms (DEBUG line of the last render).

| size | render 1 | render 2 | render 3 |
|---|---|---|---|
| 424x562 | 247 ms | 234 ms | 227 ms |
| 1920x1080 | 600 ms | 559 ms | 559 ms |
| 3840x2160 | 1669 ms | 1560 ms | 1547 ms |
| 4096x2304 | 2037 ms | 1851 ms | 1749 ms |

Per-step ms (plugin DEBUG lines, last render of each size):

```
render t=2 424x562 scale=1 view=0 | hairLocks rev 1 jumpRepair=off 0.0ms 0 regionWeight=ran 2.0ms 0 model=off 0.0ms 0 tracer=ran 4.2ms 535 detail=off 0.0ms 0 strandVote=ran 200.9ms 3 fusion=ran 0.2ms 532 snapRegion=ran 9.2ms 1261 temporal=ran 3.3ms 323 | total 220.1ms
render t=2 1920x1080 scale=1 view=0 | hairLocks rev 1 jumpRepair=off 0.0ms 0 regionWeight=ran 16.3ms 0 model=off 0.0ms 0 tracer=ran 6.7ms 1038 detail=off 0.0ms 0 strandVote=ran 393.0ms 3 fusion=ran 1.4ms 1029 snapRegion=ran 33.4ms 3659 temporal=ran 11.6ms 1011 | total 465.3ms
render t=2 3840x2160 scale=1 view=0 | hairLocks rev 1 jumpRepair=off 0.0ms 0 regionWeight=ran 78.8ms 0 model=off 0.0ms 0 tracer=ran 13.6ms 2053 detail=off 0.0ms 0 strandVote=ran 657.6ms 3 fusion=ran 5.5ms 2050 snapRegion=ran 138.3ms 9989 temporal=ran 47.0ms 1868 | total 950.7ms
render t=2 4096x2304 scale=1 view=0 | hairLocks rev 1 jumpRepair=off 0.0ms 0 regionWeight=ran 85.0ms 0 model=off 0.0ms 0 tracer=ran 14.4ms 2175 detail=off 0.0ms 0 strandVote=ran 684.5ms 3 fusion=ran 4.2ms 2167 snapRegion=ran 166.2ms 10834 temporal=ran 57.0ms 1836 | total 1022.4ms
```

Vote lines (one per render):

```
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 192.8 trace 19.3 vote 2.6 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 185.5 trace 19.5 vote 2.6 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 179.0 trace 19.3 vote 2.6 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 364.3 trace 29.9 vote 5.7 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 358.4 trace 29.8 vote 5.7 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 357.6 trace 29.7 vote 5.7 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 584.7 trace 58.2 vote 18.6 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 577.1 trace 57.5 vote 18.6 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 581.0 trace 57.8 vote 18.7 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 682.3 trace 62.0 vote 21.3 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 693.6 trace 62.4 vote 21.3 ms | Temporal used the shared flows
render t=2 vote: neighbours traced 4 judged 4 | kept 3 dropped 0 reinstated 0 | flows 601.2 trace 62.0 vote 21.3 ms | Temporal used the shared flows
```
