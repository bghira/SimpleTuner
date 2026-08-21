# MixFlow Training

MixFlow flow-matching models के लिए post-training method है। यह timestep $t$ पर model को अधिक noisy ground-truth interpolation देता है। इससे training की exact interpolation और sampling के imperfect latent के बीच का अंतर घटता है।

## Configuration

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` slowed-interpolation range नियंत्रित करता है। `0.8` paper default है। `0.0` standard interpolation रखता है, लेकिन MixFlow timestep sampling का उपयोग करता है।

MixFlow data-ward model timestep को $Beta(2,1)$ से sample करता है। SimpleTuner flow sigma को उलटी noise-ward दिशा में रखता है, इसलिए implementation $sigma = 1 - sqrt(U)$ sample करता है और फिर configured flow schedule shift लगाता है। Model को original timestep मिलता है। Latent input में यह sigma उपयोग होता है:

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

Linear flow path के लिए velocity target नहीं बदलता। Inference भी नहीं बदलता।

## Support

`flow_matching` prediction type वाली सभी SimpleTuner model families shared MixFlow path उपयोग करती हैं। Model wrappers data-ward timestep conventions, nonlinear sigma transforms और joint audio/video inputs संभालते हैं।

MixFlow को custom/uniform/Beta/fast flow schedules, Self-Flow, TwinFlow, scheduled sampling या distillation के साथ उपयोग नहीं किया जा सकता। Schedule shift समर्थित है।

MixFlow को existing flow model के post-training के लिए उपयोग करें। पहले short conventional continuation वाला learning rate और optimizer रखें, फिर fixed-seed validation samples को starting checkpoint से compare करें।

## References

- [MixFlow paper](https://arxiv.org/abs/2512.19311)
- [Reference implementation](https://github.com/fudan-generative-vision/MixFlow)
