import re
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from simpletuner.helpers.models.longcat_image import calculate_shift, prepare_pos_ids, retrieve_timesteps, split_quotation
from simpletuner.helpers.models.longcat_image.pipeline_output import LongCatImagePipelineOutput
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_prompt_language(prompt):
    pattern = re.compile(r"[\u4e00-\u9fff]")
    return "zh" if bool(pattern.search(prompt)) else "en"


SYSTEM_PROMPT_EN = """
You are a prompt engineering expert for text-to-image models. Since text-to-image models have limited capabilities in understanding user prompts, you need to identify the core theme and intent of the user's input and improve the model's understanding accuracy and generation quality through optimization and rewriting. The rewrite must strictly retain all information from the user's original prompt without deleting or distorting any details.
Specific requirements are as follows:
1. The rewrite must not affect any information expressed in the user's original prompt; the rewritten prompt should use coherent natural language, avoid low-information redundant descriptions, and keep the rewritten prompt length as concise as possible.
2. Ensure consistency between input and output languages: Chinese input yields Chinese output, and English input yields English output. The rewritten token count should not exceed 512.
3. The rewritten description should further refine subject characteristics and aesthetic techniques appearing in the original prompt, such as lighting and textures.
4. If the original prompt does not specify an image style, ensure the rewritten prompt uses a **realistic photography style**. If the user specifies a style, retain the user's style.
5. When the original prompt requires reasoning to clarify user intent, use logical reasoning based on world knowledge to convert vague abstract descriptions into specific tangible objects (e.g., convert "the tallest animal" to "a giraffe").
6. When the original prompt requires text generation, please use double quotes to enclose the text part (e.g., `"50% OFF"`).
7. When the original prompt requires generating text-heavy scenes like webpages, logos, UIs, or posters, and no specific text content is specified, you need to infer appropriate text content and enclose it in double quotes. For example, if the user inputs: "A tourism flyer with a grassland theme," it should be rewritten as: "A tourism flyer with the image title 'Grassland'."
8. When negative words exist in the original prompt, ensure the rewritten prompt does not contain negative words. For example, "a lakeside without boats" should be rewritten such that the word "boat" does not appear at all.
9. Except for text content explicitly requested by the user, **adding any extra text content is prohibited**.
Here are examples of rewrites for different types of prompts:
# Examples (Few-Shot Learning)
  1. User Input: An animal with nine lives.
    Rewrite Output: A cat bathed in soft sunlight, its fur soft and glossy. The background is a comfortable home environment with light from the window filtering through curtains, creating a warm light and shadow effect. The shot uses a medium distance perspective to highlight the cat's leisurely and stretched posture. Light cleverly hits the cat's face, emphasizing its spirited eyes and delicate whiskers, adding depth and affinity to the image.
  2. User Input: Create an anime-style tourism flyer with a grassland theme.
    Rewrite Output: In the lower right of the center, a short-haired girl sits sideways on a gray, irregularly shaped rock. She wears a white short-sleeved dress and brown flat shoes, holding a bunch of small white flowers in her left hand, smiling with her legs hanging naturally. The girl has dark brown shoulder-length hair with bangs covering her forehead, brown eyes, and a slightly open mouth. The rock surface has textures of varying depths. To the girl's left and front is lush grass, with long, yellow-green blades, some glowing golden in the sunlight. The grass extends into the distance, forming rolling green hills that fade in color as they recede. The sky occupies the upper half of the picture, pale blue dotted with a few fluffy white clouds. In the upper left corner, there is a line of text in italic, dark green font reading "Explore Nature's Peace". Colors are dominated by green, blue, and yellow, fluid lines, and distinct light and shadow contrast, creating a quiet and comfortable atmosphere.
  3. User Input: A Christmas sale poster with a red background, promoting a Buy 1 Get 1 Free milk tea offer.
    Rewrite Output: The poster features an overall red tone, embellished with white snowflake patterns on the top and left side. The upper right features a bunch of holly leaves with red berries and a pine cone. In the upper center, golden 3D text reads "Christmas Heartwarming Feedback" centered, along with red bold text "Buy 1 Get 1". Below, two transparent cups filled with bubble tea are placed side by side; the tea is light brown with dark brown pearls scattered at the bottom and middle. Below the cups, white snow piles up, decorated with pine branches, red berries, and pine cones. A blurry Christmas tree is faintly visible in the lower right corner. The image has high clarity, accurate text content, a unified design style, a prominent Christmas theme, and a reasonable layout, providing strong visual appeal.
  4. User Input: A woman indoors shot in natural light, smiling with arms crossed, showing a relaxed and confident posture.
    Rewrite Output: The image features a young Asian woman with long dark brown hair naturally falling over her shoulders, with some strands illuminated by light, showing a soft sheen. Her features are delicate, with long eyebrows, bright and spirited dark brown eyes looking directly at the camera, revealing peace and confidence. She has a high nose bridge, full lips with nude lipstick, and corners of the mouth slightly raised in a faint smile. Her skin is fair, with cheeks and collarbones illuminated by warm light, showing a healthy ruddiness. She wears a black spaghetti strap tank top revealing graceful collarbone lines, and a thin gold necklace with small beads and metal bars glinting in the light. Her outer layer is a beige knitted cardigan, soft in texture with visible knitting patterns on the sleeves. Her arms are crossed over her chest, hands covered by the cardigan sleeves, in a relaxed posture. The background is a pure dark brown without extra decoration, making the figure the absolute focus. The figure is located in the center of the frame. Light enters from the upper right, creating bright spots on her left cheek, neck, and collarbone, while the right side is slightly shadowed, creating a three-dimensional and soft tone. Image details are clear, showcasing skin texture, hair, and clothing materials well. Colors are dominated by warm tones, with the combination of beige and dark brown creating a warm and comfortable atmosphere. The overall style is natural, elegant, and artistic.
  5. User Input: Create a series of images showing the growth process of an apple from seed to fruit. The series should include four stages: 1. Sowing, 2. Seedling growth, 3. Plant maturity, 4. Fruit harvesting.
    Rewrite Output: A 4-panel exquisite illustration depicting the growth process of an apple, capturing each stage precisely and clearly. 1. "Sowing": A close-up shot of a hand gently placing a small apple seed into fertile dark soil, with visible soil texture and the seed's smooth surface. The background is a soft-focus garden dotted with green leaves and sunlight filtering through. 2. "Seedling Growth": A young apple sapling breaks through the soil, stretching tender green leaves toward the sky. The scene is set in a vibrant garden illuminated by warm golden light, highlighting the seedling's delicate structure. 3. "Plant Maturity": A mature apple tree, lush with branches and leaves, covered in tender green foliage and developing small apples. The background is a vibrant orchard under a clear blue sky, with dappled sunlight creating a peaceful atmosphere. 4. "Fruit Harvesting": A hand reaches into the tree to pick a ripe red apple, its smooth skin glistening in the sun. The scene shows the abundance of the orchard, with baskets of apples in the background, giving a sense of fulfillment. Each illustration uses a realistic style, focusing on details and harmonious colors to showcase the natural beauty and development of the apple's life cycle.
  6. User Input: If 1 represents red, 2 represents green, 3 represents purple, and 4 represents yellow, please generate a four-color rainbow based on this rule. The color order from top to bottom is 3142.
    Rewrite Output: The image consists of four horizontally arranged colored stripes, ordered from top to bottom as purple, red, yellow, and green. A white number is centered on each stripe. The top purple stripe features the number "3", the red stripe below it has the number "1", the yellow stripe further down has the number "4", and the bottom green stripe has the number "2". All numbers use a sans-serif font in pure white, forming a sharp contrast with the background colors to ensure good readability. The stripes have high color saturation and a slight texture. The overall layout is simple and clear, with distinct visual effects and no extra decorative elements, emphasizing the numerical information. The image is high definition, with accurate colors and a consistent style, offering strong visual appeal.
  7. User Input: A stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", natural light, background is a Chinese garden.
    Rewrite Output: An ancient stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", the surface covered with traces of time, the writing clear and deep. Natural light falls from above, softly illuminating every detail of the stone tablet and enhancing its sense of history. The background is an elegant Chinese garden featuring lush bamboo forests, winding paths, and quiet pools, creating a serene and distant atmosphere. The overall picture uses a realistic style with rich details and natural light and shadow effects, highlighting the cultural heritage of the stone tablet and the classical beauty of the garden.
# Output Format
Please directly output the rewritten and optimized Prompt content. Do not include any explanatory language or JSON formatting, and do not add opening or closing quotes yourself."""


SYSTEM_PROMPT_ZH = """
你是一名文生图模型的prompt engineering专家。由于文生图模型对用户prompt的理解能力有限，你需要识别用户输入的核心主题和意图，并通过优化改写提升模型的理解准确性和生成质量。改写必须严格保留用户原始prompt的所有信息，不得删减或曲解任何细节。
具体要求如下：
1. 改写不能影响用户原始prompt里表达的任何信息，改写后的prompt应该使用连贯的自然语言表达,不要出现低信息量的冗余描述，尽可能保持改写后prompt长度精简。
2. 请确保输入和输出的语言类型一致，中文输入中文输出，英文输入英文输出，改写后的token数量不要超过512个;
3. 改写后的描述应当进一步完善原始prompt中出现的主体特征、美学技巧，如打光、纹理等；
4. 如果原始prompt没有指定图片风格时，确保改写后的prompt使用真实摄影风格，如果用户指定了图片风格，则保留用户风格；
5. 当原始prompt需要推理才能明确用户意图时，根据世界知识进行适当逻辑推理，将模糊抽象描述转化为具体指向事物（例：将"最高的动物"转化为"一头长颈鹿"）。
6. 当原始prompt需要生成文字时，请使用双引号圈定文字部分，例：`"限时5折"`）。
7. 当原始prompt需要生成网页、logo、ui、海报等文字场景时，且没有指定具体的文字内容时，需要推断出合适的文字内容，并使用双引号圈定，如用户输入：一个旅游宣传单，以草原为主题。应该改写成：一个旅游宣传单，图片标题为“草原”。
8. 当原始prompt中存在否定词时，需要确保改写后的prompt不存在否定词，如没有船的湖边，改写后的prompt不能出现船这个词汇。
9. 除非用户指定生成品牌logo，否则不要增加额外的品牌logo.
10. 除了用户明确要求书写的文字内容外，**禁止增加任何额外的文字内容**。
以下是针对不同类型prompt改写的示例：

# Examples (Few-Shot Learning)
  1. 用户输入: 九条命的动物。
    改写输出: 一只猫，被柔和的阳光笼罩着，毛发柔软而富有光泽。背景是一个舒适的家居环境，窗外的光线透过窗帘，形成温馨的光影效果。镜头采用中距离视角，突出猫悠闲舒展的姿态。光线巧妙地打在猫的脸部，强调它灵动的眼睛和精致的胡须，增加画面的层次感与亲和力。
  2. 用户输入: 制作一个动画风格的旅游宣传单，以草原为主题。
    改写输出: 画面中央偏右下角，一个短发女孩侧身坐在灰色的不规则形状岩石上，她穿着白色短袖连衣裙和棕色平底鞋，左手拿着一束白色小花，面带微笑，双腿自然垂下。女孩的头发为深棕色，齐肩短发，刘海覆盖额头，眼睛呈棕色，嘴巴微张。岩石表面有深浅不一的纹理。女孩的左侧和前方是茂盛的草地，草叶细长，呈黄绿色，部分草叶在阳光下泛着金色的光芒，仿佛被阳光照亮。草地向远处延伸，形成连绵起伏的绿色山丘，山丘的颜色由近及远逐渐变浅。天空占据了画面的上半部分，呈淡蓝色，点缀着几朵白色蓬松的云彩。画面的左上角有一行文字，文字内容是斜体、深绿色的“Explore Nature's Peace”。色彩以绿色、蓝色和黄色为主，线条流畅，光影明暗对比明显，营造出一种宁静、舒适的氛围。
  3. 用户输入: 一张以红色为背景的圣诞节促销海报，主要宣传奶茶买一送一的优惠活动。
    改写输出: 海报整体呈现红色调，上方和左侧点缀着白色雪花图案，右上方有一束冬青叶和红色浆果，以及一个松果。海报中央偏上位置，金色立体字样“圣诞节 暖心回馈”居中排列，和红色粗体字“买1送1”。海报下方，两个装满珍珠奶茶的透明杯子并排摆放，杯中奶茶呈浅棕色，底部和中间散布着深棕色珍珠。杯子下方，堆积着白色雪花，雪花上装饰着松枝、红色浆果和松果。右下角隐约可见一棵模糊的圣诞树。图片清晰度高，文字内容准确，整体设计风格统一，圣诞主题突出，排版布局合理，具有较强的视觉吸引力。
  4. 用户输入: 一位女性在室内以自然光线拍摄，她面带微笑，双臂交叉，展现出轻松自信的姿态。
    改写输出: 画面中是一位年轻的亚洲女性，她拥有深棕色的长发，发丝自然地垂落在双肩，部分发丝被光线照亮，呈现出柔和的光泽。她的五官精致，眉毛修长，眼睛明亮有神，瞳孔呈深棕色，眼神直视镜头，流露出平和与自信。鼻梁挺拔，嘴唇丰满，涂有裸色系唇膏，嘴角微微上扬，展现出浅浅的微笑。她的肤色白皙，脸颊和锁骨处被暖色调的光线照亮，呈现出健康的红润感。她穿着一件黑色的细吊带背心，肩带纤细，露出优美的锁骨线条。脖颈上佩戴着一条金色的细项链，项链由小珠子和几个细长的金属条组成，在光线下闪烁着光泽。她的外搭是一件米黄色的针织开衫，材质柔软，袖子部分有明显的针织纹理。她双臂交叉在胸前，双手被开衫的袖子覆盖，姿态放松。背景是纯粹的深棕色，没有多余的装饰，使得人物成为画面的绝对焦点。人物位于画面中央。光线从画面的右上方射入，在人物的左侧脸颊、脖颈和锁骨处形成明亮的光斑，右侧则略显阴影，营造出立体感和柔和的影调。图像细节清晰，人物的皮肤纹理、发丝以及衣物材质都得到了很好的展现。色彩以暖色调为主，米黄色和深棕色的搭配营造出温馨舒适的氛围。整体呈现出一种自然、优雅且富有亲和力的艺术风格。
  5. 用户输入：创作一系列图片，展现苹果从种子到结果的生长过程。该系列图片应包含以下四个阶段：1. 播种，2. 幼苗生长，3. 植物成熟，4. 果实采摘。
    改写输出：一个4宫格的精美插图，描绘苹果的生长过程，精确清晰地捕捉每个阶段。1.“播种”：特写镜头，一只手轻轻地将一颗小小的苹果种子放入肥沃的深色土壤中，土壤的纹理和种子光滑的表面清晰可见。背景是花园的柔焦画面，点缀着绿色的树叶和透过树叶洒下的阳光。2.“幼苗生长”：一棵幼小的苹果树苗破土而出，嫩绿的叶子向天空舒展。场景设定在一个生机勃勃的花园中，温暖的金光照亮了它。幼苗的纤细结构。3.“植物的成熟”：一棵成熟的苹果树，枝繁叶茂，挂满了嫩绿的叶子和正在萌发的小苹果。背景是一片生机勃勃的果园，湛蓝的天空下，斑驳的阳光营造出宁静祥和的氛围。4.“采摘果实”：一只手伸向树上，摘下一个成熟的红苹果，苹果光滑的果皮在阳光下闪闪发光。画面展现了果园的丰收景象，背景中摆放着一篮篮的苹果，给人一种圆满满足的感觉。每幅插图都采用写实风格，注重细节，色彩和谐，展现了苹果生命周期的自然之美和发展过程。
  6. 用户输入： 如果1代表红色，2代表绿色，3代表紫色，4代表黄色，请按照此规则生成四色彩虹。它的颜色顺序从上到下是3142
    改写输出：图片由四个水平排列的彩色条纹组成，从上到下依次为紫色、红色、黄色和绿色。每个条纹上都居中放置一个白色数字。最上方的紫色条纹上是数字“3”，其下方红色条纹上是数字“1”，再下方黄色条纹上是数字“4”，最下方的绿色条纹上是数字“2”。所有数字均采用无衬线字体，颜色为纯白色，与背景色形成鲜明对比，确保了良好的可读性。条纹的颜色饱和度高，且带有轻微的纹理感，整体排版简洁明了，视觉效果清晰，没有多余的装饰元素，强调了数字信息本身。图片整体清晰度高，色彩准确，风格一致，具有较强的视觉吸引力。
  7. 用户输入：石碑上刻着“关关雎鸠，在河之洲”，自然光照，背景是中式园林
    改写输出：一块古老的石碑上刻着“关关雎鸠，在河之洲”，石碑表面布满岁月的痕迹，字迹清晰而深刻。自然光线从上方洒下，柔和地照亮石碑的每一个细节，增强了其历史感。背景是一座典雅的中式园林，园林中有翠绿的竹林、蜿蜒的小径和静谧的水池，营造出一种宁静而悠远的氛围。整体画面采用写实风格，细节丰富，光影效果自然，突出了石碑的文化底蕴和园林的古典美。
# 输出格式
请直接输出改写优化后的 Prompt 内容，不要包含任何解释性语言或 JSON 格式，不要自行添加开头或结尾的引号。
"""


class LongCatImagePipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    Pipeline for LongCat-Image text-to-image generation.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor", "text_processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        text_processor: AutoProcessor,
        transformer,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            text_processor=text_processor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 36
        self.prompt_template_encode_end_idx = 5
        self.default_sample_size = 128
        self.max_tokenizer_len = 512

    @torch.inference_mode()
    def rewire_prompt(self, prompt, device):
        language = get_prompt_language(prompt)
        if language == "zh":
            question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{prompt}\n改写后的prompt为："
        else:
            question = SYSTEM_PROMPT_EN + f"\nUser Input: {prompt}\nRewritten prompt:"

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        text = self.text_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.text_processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(device)

        generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=self.max_tokenizer_len)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.text_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    @torch.inference_mode()
    def encode_prompt(self, prompts, device, dtype):
        prompts = [
            prompt.strip('"') if isinstance(prompt, str) and prompt.startswith('"') and prompt.endswith('"') else prompt
            for prompt in prompts
        ]
        all_tokens = []
        for clean_prompt_sub, matched in split_quotation(prompts[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)

        all_tokens = all_tokens[: self.max_tokenizer_len]
        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": [all_tokens]},
            max_length=self.max_tokenizer_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        prefix_tokens = self.tokenizer(self.prompt_template_encode_prefix, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat((prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, self.prompt_template_encode_start_idx : -self.prompt_template_encode_end_idx, :]

        dtype_pos_ids = dtype
        if device.type == "mps" and dtype_pos_ids == torch.float64:
            dtype_pos_ids = torch.float32
        text_ids = prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(device, dtype=dtype_pos_ids)

        return prompt_embeds, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        prompt_length: Optional[int] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        dtype_pos = torch.float64 if device.type != "mps" else torch.float32
        text_token_count = prompt_length if prompt_length is not None else self.max_tokenizer_len
        latent_image_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(text_token_count, text_token_count),
            height=height // 2,
            width=width // 2,
        ).to(device, dtype=dtype_pos)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device)
        latents = latents.to(dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents, latent_image_ids

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        text_ids: Optional[torch.FloatTensor] = None,
        negative_text_ids: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        enable_cfg_renorm: Optional[bool] = True,
        cfg_renorm_min: Optional[float] = 0.0,
        enable_prompt_rewrite: Optional[bool] = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )
            pixel_step = self.vae_scale_factor * 2
            height = int(height / pixel_step) * pixel_step
            width = int(width / pixel_step) * pixel_step

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("A prompt or precomputed prompt_embeds must be provided.")

        device = self._execution_device
        prompt_length = prompt_embeds.shape[1] if prompt_embeds is not None else None

        if prompt_embeds is None:
            if enable_prompt_rewrite and prompt is not None:
                prompt = self.rewire_prompt(prompt, device)
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = [negative_prompt] * num_images_per_prompt
            prompt = [prompt] * num_images_per_prompt if prompt is not None else [""]

            prompt_embeds, text_ids = self.encode_prompt(prompt, device, dtype=torch.float64)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(negative_prompt, device, dtype=torch.float64)
            prompt_length = prompt_embeds.shape[1]
        else:
            if text_ids is None:
                text_ids = prepare_pos_ids(
                    modality_id=0,
                    type="text",
                    start=(0, 0),
                    num_token=prompt_length,
                ).to(device, dtype=torch.float64)
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_text_ids = negative_text_ids.to(device) if negative_text_ids is not None else text_ids
            prompt_length = prompt_embeds.shape[1]

        num_channels_latents = 16
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            prompt_length=prompt_length,
        )

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        guidance = None
        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                neg_prompt = negative_prompt
                if neg_prompt is None:
                    neg_prompt = [""] * batch_size
                elif isinstance(neg_prompt, str):
                    neg_prompt = [neg_prompt] * batch_size
                neg_dtype = prompt_embeds.dtype if prompt_embeds is not None else torch.float32
                if device.type == "mps" and neg_dtype == torch.float64:
                    neg_dtype = torch.float32
                negative_prompt_embeds, negative_text_ids = self.encode_prompt(neg_prompt, device, dtype=neg_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
        else:
            prompt_embeds = prompt_embeds.to(device)

        text_ids = text_ids.to(device) if text_ids is not None else None
        latent_image_ids = latent_image_ids.to(device)

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Set begin index to avoid timestep lookup in scheduler.step() which can fail due to
        # floating-point precision issues. See: https://github.com/huggingface/diffusers/pull/11696
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                _tlora_cfg = getattr(self, "_tlora_config", None)
                if _tlora_cfg:
                    apply_tlora_inference_mask(
                        timestep=int(t),
                        max_timestep=self.scheduler.config.num_train_timesteps,
                        max_rank=_tlora_cfg["max_rank"],
                        min_rank=_tlora_cfg["min_rank"],
                        alpha=_tlora_cfg["alpha"],
                    )
                try:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                finally:
                    if _tlora_cfg:
                        clear_tlora_mask()

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if enable_cfg_renorm:
                        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        noise_pred = noise_pred * scale

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    # Provide unpacked latents for preview decoders while keeping the packed version for inference.
                    latents_for_callback = self._unpack_latents(
                        latents,
                        height,
                        width,
                        self.vae_scale_factor,
                    )
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                            callback_kwargs[k] = latents_for_callback
                        elif k == "latents_packed":
                            callback_kwargs[k] = latents
                        else:
                            callback_kwargs[k] = locals().get(k, None)
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    cb_latents = callback_outputs.pop("latents", None)
                    if cb_latents is not None:
                        if cb_latents.dim() == 4:
                            latents = self._pack_latents(
                                cb_latents,
                                batch_size=cb_latents.shape[0],
                                num_channels_latents=cb_latents.shape[1],
                                height=cb_latents.shape[2],
                                width=cb_latents.shape[3],
                            ).to(device=latents.device, dtype=latents.dtype)
                        elif cb_latents.dim() == 3:
                            latents = cb_latents.to(device=latents.device, dtype=latents.dtype)
                    if callback_outputs:
                        logger.debug(
                            "Callback returned additional tensors that were ignored: %s",
                            list(callback_outputs.keys()),
                        )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return LongCatImagePipelineOutput(images=image)
