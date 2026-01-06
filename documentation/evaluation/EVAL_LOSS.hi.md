SimpleTuner की एक प्रयोगात्मक सुविधा ["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) में दिए विचारों को लागू करती है ताकि मूल्यांकन के लिए एक स्थिर loss मान प्राप्त किया जा सके।

इसके प्रयोगात्मक होने के कारण, यह कुछ समस्याएँ पैदा कर सकती है या वह कार्यक्षमता/इंटीग्रेशन न हो जो पूरी तरह अंतिम फीचर में होगा।

इसे प्रोडक्शन में इस्तेमाल करना ठीक है, लेकिन भविष्य के संस्करणों में बग या बदलाव की संभावना का ध्यान रखें।

उदाहरण डाटालोडर:

```json
[
    {
        "id": "something-special-to-remember-by",
        "crop": false,
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 512,
        "maximum_image_size": 1536,
        "target_downsample_size": 512,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/sana",
        "vae_cache_clear_each_epoch": false,
        "skip_file_discovery": ""
    },
    {
        "id": "sana-eval",
        "type": "local",
        "dataset_type": "eval",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/sana-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sana"
    }
]
```

- Eval इमेज डेटासेट को सामान्य इमेज डेटासेट की तरह ही कॉन्फ़िगर किया जा सकता है।
- मूल्यांकन डेटासेट ट्रेनिंग में **उपयोग नहीं** होता।
- सुझाव है कि ऐसी इमेजेस उपयोग करें जो आपके ट्रेनिंग सेट से बाहर के कॉन्सेप्ट्स को दर्शाती हों।

Evaluation loss गणना कॉन्फ़िगर और सक्षम करने के लिए:

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb"
}
```

अब evaluations को step या epoch के अनुसार शेड्यूल किया जा सकता है। `--eval_epoch_interval` दशमलव मान स्वीकार करता है, इसलिए `0.5`
प्रति epoch दो बार evaluation चलाएगा। यदि आप `--eval_steps_interval` और `--eval_epoch_interval` दोनों सेट करते हैं, तो
trainer एक चेतावनी लॉग करेगा और दोनों शेड्यूल पर evaluations चलाएगा।

Evaluation loss गणना को बंद करने के लिए, लेकिन eval datasets कॉन्फ़िगर रखें (उदाहरण के लिए केवल CLIP स्कोरिंग के लिए):

```json
{
    "--eval_loss_disable": true
}
```

यह तब उपयोगी है जब आप eval datasets को CLIP स्कोर मीट्रिक्स (`--evaluation_type clip`) के लिए उपयोग करना चाहते हैं बिना हर timestep पर validation loss की गणना के ओवरहेड के।

> **नोट**: पूर्ण evaluation चार्टिंग कार्यक्षमता के लिए Weights & Biases (wandb) फिलहाल आवश्यक है। अन्य trackers को केवल एकल mean मान मिलता है।
