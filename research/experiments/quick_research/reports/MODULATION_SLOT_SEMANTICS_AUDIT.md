# Modulation slot semantics audit

Source: `636ca0ef517a4db087a6a08a6a8a5e704e21f836`. Build relationship: different.

Version boundary: Findings apply directly to mtytel/vital@636ca0ef517a4db087a6a08a6a8a5e704e21f836, whose plugin project reports version 1.0.6. Every cited implementation block is unchanged in DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76. Fifteen of sixteen compared core files are byte-identical; synth_parameters.cpp differs only in unrelated destination-range and lookup code, leaving its modulation parameter block unchanged. Vita's synth_base additions are wrapper/rendering changes outside the cited routing paths. This comparison does not establish equivalence to the installed Vital 1.6.4 VST3.

Build evidence: The audited mtytel/vital source declares plugin version 1.0.6 in plugin/JuceLibraryCode/JuceHeader.h, while experiments B/C used the separately installed Vital 1.6.4 VST3. No VST3 commit identifier is available.

| Route/dependency case | Evaluation semantics | Required remapping | Guarantee and version coverage |
|---|---|---|---|
| Independent acyclic routes | Each positional connection uses the same-index keyed controls and optional line map. | Move the connection object, line map, and all five keyed controls; rewrite every slot-amount reference. | Semantic equivalence from equivalent state in the pinned Vital/Vita sources; no bit-identity guarantee. |
| Multiple routes to one destination | Contributors are summed in connection/input order after dependency scheduling. | Preserve contributor order for any numerical reproducibility claim. | The mathematical sum is equivalent, but sequential floating-point accumulation can differ. |
| Acyclic route-amount dependency | The producer is scheduled before the consumer through the normal dependency graph. | Rewrite the destination to the relabeled amount control and move both slots completely. | Semantic equivalence holds only for the complete graph-aware relabel. |
| Cyclic route-amount dependency | The edge that closes the cycle receives a previous-block feedback node. | Preserve construction order and which edge is delayed, in addition to full slot remapping. | Reordering can change semantics; invariance is not established. |
| Installed Vital 1.6.4 VST3 | Its source commit was not identified. | Preserve stored order and dependency structure. | The pinned-source conclusions are version-limited for this binary. |

## loading: supported

How do connection arrays and all modulation_<slot>_* controls map to runtime slots?

Vital owns a fixed bank of 64 ModulationConnection objects. Array index i is runtime slot i+1: saving emits all 64 positional objects, and loading walks the JSON array while binding element i to bank.atIndex(i). The connection object carries source, destination, processor, and optional line mapping. Its keyed controls are modulation_{i+1}_amount, _power, _bipolar, _stereo, and _bypass: the voice handler wires amount and power, while the indexed processor creates the three flags. The positional object, including its optional line mapping, and those five keyed settings therefore form one slot. Permuting array objects without moving the keyed controls changes which controls drive each route; editing only source/destination within an object also leaves that slot's existing line mapping attached.

- [src/common/synth_constants.h:28–38](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/synth_constants.h#L28-L38)
- [src/common/synth_types.cpp:37–66](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/synth_types.cpp#L37-L66)
- [src/common/load_save.cpp:94–119](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/load_save.cpp#L94-L119)
- [src/common/load_save.cpp:170–194](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/load_save.cpp#L170-L194)
- [src/synthesis/modules/synth_voice_handler.cpp:79–95](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/synth_voice_handler.cpp#L79-L95)
- [src/synthesis/modules/modulation_connection_processor.cpp:39–49](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L39-L49)

## graph: supported

How are route traversal, dependency scheduling and mono/poly accumulation ordered?

On connection, the engine plugs the source into the route processor, chooses the polyphonic destination only when the source is polyphonic and the destination has a polyphonic counterpart, and appends the route output to that destination. Modulated controls are explicit sums: mono controls use VariableAdd; poly controls add a separate poly sum to the base/mono result, using ModulationSum for audio-rate controls. The router follows input dependencies and moves producers before consumers. Within a destination, contributors are accumulated in input order, and plugNext fills the first empty input before appending another. Slot number is not an explicit DSP priority, but load/connect order determines sum-input order while graph dependencies determine processor execution order.

- [src/synthesis/synth_engine/sound_engine.cpp:139–167](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/synth_engine/sound_engine.cpp#L139-L167)
- [src/synthesis/framework/synth_module.cpp:54–85](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/synth_module.cpp#L54-L85)
- [src/synthesis/framework/synth_module.cpp:146–188](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/synth_module.cpp#L146-L188)
- [src/synthesis/framework/processor.cpp:105–121](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/processor.cpp#L105-L121)
- [src/synthesis/framework/operators.cpp:187–250](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/operators.cpp#L187-L250)
- [src/synthesis/framework/processor_router.cpp:209–244](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/processor_router.cpp#L209-L244)

## state: version_limited

What smoothing, ramp and initialization/reset state belongs to each route, and in which versions?

A route starts at control rate with zero cached amount, power, and destination scale, plus a linear line map. A destination-scale change zeros the cached amount. The control-rate path applies mapping, bipolar/power/stereo shaping, amount, and destination scale immediately. The audio-rate paths retain amount and power across processing blocks and linearly ramp each from its cached value to the current target over the block; a reset mask snaps the starting state to the target. Bypass returns zero before this work. Audio-rate ModulationSum separately retains the preceding control contribution and ramps it across the block. Control-rate feedback used for cycles retains the preceding block value and resets to zero. These facts are supported for both audited commits, but source equivalence to the installed Vital 1.6.4 VST3 remains unestablished.

- [src/synthesis/modules/modulation_connection_processor.cpp:22–64](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L22-L64)
- [src/synthesis/modules/modulation_connection_processor.cpp:67–110](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L67-L110)
- [src/synthesis/modules/modulation_connection_processor.cpp:112–152](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L112-L152)
- [src/synthesis/modules/modulation_connection_processor.cpp:245–285](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L245-L285)
- [src/synthesis/framework/operators.cpp:216–250](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/operators.cpp#L216-L250)
- [src/synthesis/framework/feedback.h:48–65](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/feedback.h#L48-L65)

## dependencies: exception

What happens when a route targets another route's amount, including cycles and relabeling?

createConnection scans the bank and refuses to place a route in slot i when its destination is that same slot's modulation_i_amount; it may place that destination in another free slot. A cross-slot amount route then participates in the normal dependency graph. When a newly connected edge would close a cycle, the router substitutes a control-rate Feedback node, whose output is the preceding processing block's value and whose reset value is zero. The JSON loader binds bank slots directly rather than calling createConnection, so arbitrary external JSON cannot be assumed to retain the allocator's self-target rule. A semantics-preserving slot relabel must move the positional route entry and line map, move all five keyed slot controls, and rewrite every modulation_j_amount destination reference. For cycles, changing connection order can change which edge closes the cycle and receives the block delay.

- [src/common/synth_types.cpp:50–70](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/synth_types.cpp#L50-L70)
- [src/common/load_save.cpp:170–194](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/load_save.cpp#L170-L194)
- [src/synthesis/framework/processor_router.cpp:179–207](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/processor_router.cpp#L179-L207)
- [src/synthesis/framework/feedback.h:48–65](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/feedback.h#L48-L65)

## numerics: exception

Which guarantees are semantic, numerical-with-tolerance, or bit-identical, and what exceptions apply?

For an acyclic graph, a complete relabel that moves every per-slot field and rewrites every slot-amount reference preserves the mathematical routing function when it starts from equivalent state. The code does not guarantee bit-identical output: contributors are added sequentially in connection/input order, floating-point addition is not associative, and audio-rate routes carry block-ramped state. Cyclic graphs add a block delay on the edge selected when the cycle is closed, so reordered connection construction can change semantics as well as numerics. The defensible code-only claim is semantic equivalence under the stated acyclic/full-relabel/same-state conditions; numerical equality should use a tolerance, and its size is not established by source inspection.

- [src/synthesis/framework/processor.cpp:105–121](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/processor.cpp#L105-L121)
- [src/synthesis/framework/operators.cpp:187–250](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/operators.cpp#L187-L250)
- [src/synthesis/modules/modulation_connection_processor.cpp:90–110](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L90-L110)
- [src/synthesis/modules/modulation_connection_processor.cpp:112–152](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/modules/modulation_connection_processor.cpp#L112-L152)
- [src/synthesis/framework/processor_router.cpp:179–190](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/processor_router.cpp#L179-L190)

Citation existence/ranges were checked mechanically. Interpretations are reviewer-authored; this tool does not prove them. Semantic interchangeability, numerical tolerance, and bit identity must remain distinct. No permutation renders were run.
