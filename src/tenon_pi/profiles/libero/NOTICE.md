# pi05_libero profile provenance

- OpenPI revision: `215abfb217dbac7d5f1273282331b9b1866c0479`
- Released checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Released normalization asset:
  `gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`
- Normalization file SHA-256:
  `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`
- Canonical transform descriptor SHA-256:
  `81bd9293ace15b1a9209ddf514c79a88e1bf0c2a7b521a13b2449747644c20d6`
- Raw descriptor file SHA-256:
  `4157e65cc7d6ba97b585ac8ae5c63a21d4624f7322c865f83883eb1ca0cae58b`
- Public projection reference commit:
  `tenon-pi-rtc@933f91496d2adb15fd591a3f8a04c950e364eab9`
- Dataset provenance:
  `physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f`

The normalization file is copied byte-for-byte from the released OpenPI
checkpoint asset. For native `tenon-control-v1`, the public descriptor declares
an explicit componentwise `[-1,1]` physical output projection after
unnormalization/dimension trimming, matching robosuite 1.4.1 controller
saturation. `openpi-ws-v1` preserves raw upstream OpenPI actions and its upstream
client/actuator retains bounds ownership. Raw model/sampler outputs remain
unchanged and are recorded separately from native projected/applied commands.

OpenPI code is Apache-2.0 and also ships Gemma terms for model use. The LIBERO
dataset revision declares CC-BY-4.0. This client does not redistribute model
weights.
