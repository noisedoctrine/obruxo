# Let clipped controls learn back into range

The saved shared-model fit contains a concrete gradient failure: a continuous prediction can move outside its legal range, be clipped for rendering, then receive no learning signal even when the measured sound gradient points back into range. This is separate from the unfinished timbre/performance separation check and does not explain the whole reconstruction failure.

The [read-only diagnosis](data/shared_basic_learning_diagnosis_summary.json) exactly replays both models, Adam states and RNG from the previous fit. Of 1,470 active control observations, 54 proposals are outside the legal interval and 33 have useful inward gradients blocked by clipping. They account for 4.53% of the absolute gradient magnitude received at the rendered outputs, after the fit's observation weighting. No new rendering or experimental fitting occurs.

`InwardBasic` preserves the existing clipped forward values. Its backward pass allows a gradient to move an out-of-range value inward and blocks movement farther outward. Inside the interval and at its boundaries, the derivative remains one. This is an explicit approximate training derivative; it does not change Vital's renderer or claim an exact native derivative.

The [software qualification](data/inward_basic_bridge_summary.json) preserves all 1,616 cached predictions and cold exports, restores all 33 measured inward directions, and checks all 1,470 saved active components. Three simulated updates exactly recover model, Adam and RNG state. Both audio-input branches receive gradients. The original initializer remains unchanged, with **14 dense outputs** and no added parameters. The resulting artifact is still untrained.

Qualification takes 39.60 seconds and 0.69 GiB peak RAM, within ten minutes / 4 GiB / 64 MiB. It makes zero native requests and zero experimental optimizer updates. Four focused tests cover inward/outward directions, boundary behavior, actual re-entry under gradient descent, preserved inference and legacy export.

No new native shared-model training has used this repair. Representation probes come first; the full reconstruction gates, reserved confirmation panel and release requirements remain open.
