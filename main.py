import base64
import cv2
import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import soundfile as sf
import tempfile
import torch
import whisper
import easyocr

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from skimage import io

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoTokenizer,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPSegProcessor, 
    CLIPSegForImageSegmentation,
    pipeline,
)

from segment_anything import sam_model_registry, SamPredictor

from diffusers import DiffusionPipeline
from dotenv import load_dotenv

import tensorflow as tf
import tensorflow_hub as hub
from torchvision import transforms
from torchvision.transforms.functional import normalize

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
################################################
from object_utils import detect_and_segment_object, sam_integration_method, place_object_optimally
from video_utils import extract_frames, clip_similarity
################################################

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
################################################ 1.TAB ################################################

def all_function(input_img, query_text, background_img):
    img, x0, y0, x1, y1 = detect_and_segment_object(input_img, query_text)
    cropped_img = sam_integration_method(img, x0, y0, x1, y1)
    final_img = place_object_optimally(object_image=cropped_img, background_image= background_img, query_text= query_text)
    return final_img

################################################ 2.TAB ################################################
def clip_similarity_1(image, texts):
    """CLIP ile görüntü-metin benzerlik analizi"""
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    texts = [t.strip() for t in texts.split(",")]
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)  
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()
    return dict(sorted(zip(texts, probs), key=lambda x: x[1], reverse=True))

################################################ 3.TAB ################################################
def image_to_base64(image_path): 
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Görsel Soru-Cevap fonksiyonu
def visual_question_answering_chatbot(history, image, question):
    # Geçici olarak resmi kaydet (base64 dönüşüm için)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # LLM çağrısı
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Sistem mesajı
    system_message = SystemMessage(content="""
    Sen bir görsel soru-cevap uzmanısın. Resmi analiz edip kullanıcının sorusunu yanıtlamalısın.
    """)

    # Görseli base64'e çevir ve Gemini'ye uygun şekilde hazırla
    image_base64 = image_to_base64(image_path)
    user_message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
    ])

    # Sohbet geçmişine sistem mesajını ve kullanıcı mesajını ekle
    response = llm.invoke([system_message, user_message])
    history.append((question, response.content))
    return history, ""

################################################ 4.TAB ################################################
def image_description(image):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = pipe(image)
    return result[0]['generated_text']

################################################ 5.TAB ################################################
def extract_text_from_image(image):
    reader = easyocr.Reader(['tr', 'en'], gpu=False)
    image_np = np.array(image)  # PIL → NumPy
    results = reader.readtext(image_np)

    extracted_text = []
    for (bbox, text, prob) in results:
        extracted_text.append(f"{text} (Güven: {prob:.2f})")
    
    return "\n".join(extracted_text) if extracted_text else "Metin bulunamadı."

################################################ 6.TAB ################################################

def analyze_video(video_path):
    # 1. Frame çıkar
    frame_paths = extract_frames(video_path)

    # 2. CLIP ile skorlama
    scored_frames = clip_similarity(frame_paths)

    # 3. En iyi 5 frame + skorları
    top_5 = scored_frames[:5]
    top_5_paths = [item[0] for item in top_5]
    scores_text = "\n".join([f"{os.path.basename(p)}: {s:.4f}" for p, s in top_5])

    return top_5_paths, scores_text

################################################ 7.TAB ################################################

def transcribe_audio(audio_file, language="en"):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language=language)
        return result["text"]
    except Exception as e:
        return f"Bir hata oluştu: {e}"

################################################ 8.TAB ################################################
def classify_music_genre(input_path: str):

    if not os.path.exists(input_path):
        return "Dosya bulunamadı."

    filename, ext = os.path.splitext(input_path)
    if ext.lower() not in [".mp3", ".wav"]:
        return "Lütfen .mp3 veya .wav formatında bir dosya yükleyin."

    wav_path = input_path

    try:
        y, sr = librosa.load(input_path, sr=16000, duration=30)
        
        wav_path = filename + "_30s.wav"
        sf.write(wav_path, y, sr)
    except Exception as e:
        return f"Ses işleme hatası: {e}"

    try:
        classifier = pipeline("audio-classification", model="dima806/music_genres_classification")
        preds = classifier(wav_path)
    except Exception as e:
        return f"Model hatası: {e}"

    if os.path.exists(wav_path) and wav_path != input_path:
        os.remove(wav_path)

    if not preds:
        return "Tahmin yapılamadı."

    result = f"En olası tür: **{preds[0]['label']}** ({preds[0]['score']:.2f})\n\n"
    result += "Diğer Tahminler:\n"
    for pred in preds[1:]:
        result += f"- {pred['label']}: {pred['score']:.2f}\n"
    return result

################################################ 9.TAB ################################################

def stable_diffusion(prompt):
    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt).images[0]
    return image

################################################ 10.TAB ################################################

def load_image(image_path, image_size=(256, 256)):
    img = Image.open(image_path).convert('RGB').resize(image_size)
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def apply_style_transfer(content_img_path, style_img_path, image_size=(256, 256)):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    content_image = load_image(content_img_path, image_size)
    style_image = load_image(style_img_path, image_size)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0][0].numpy()  # [0] batch, sonra tensor → numpy

    stylized_image = (stylized_image * 255).astype(np.uint8)
    return Image.fromarray(stylized_image)


#######################################################################################################

with gr.Blocks() as demo:
    gr.Markdown("## Multimodal AI Uygulamaları")
    
    with gr.Tabs():
        with gr.TabItem("Nesne Kırpma ve Ekleme"):
            gr.Markdown("**Nesne kırpma ve arkaplana ekleme**")

            with gr.Row():  # Hepsi yatayda olacak
                input_mode = gr.Radio(
                    ["Resim Yükle", "Metin Gir"],
                    label="🔍 Giriş Tipi Seçin",
                    value="Resim Yükle"
                )

                input_img = gr.Image(
                    type="filepath",
                    label="🎯 Nesne Görseli (Yükleyin)",
                    visible=True
                )

                input_text = gr.Textbox(
                    label="✍️ Resim Linki",
                    placeholder="http.example.com/image.jpg",
                    visible=False
                )

                background_img = gr.Image(
                    type="filepath",
                    label="🏞️ Arka Plan Görseli (Yükleyin)"
                )

                query_text = gr.Textbox(
                    label="🎯 Kırpılacak Nesne",
                    placeholder="a dog.",
                    info="Buraya kırpılacak nesneyi yazın. Örn: a cat."
                )

            output_image = gr.Image(label="🖼️ Sonuç Görseli")

            submit_button = gr.Button("🚀 Uygula")

            # Giriş türüne göre gösterim değiştirme
            def toggle_input_visibility(mode):
                return (
                    gr.update(visible=(mode == "Resim Yükle")),
                    gr.update(visible=(mode == "Metin Gir"))
                )

            input_mode.change(
                fn=toggle_input_visibility,
                inputs=input_mode,
                outputs=[input_img, input_text]
            )

            # Ana fonksiyon sarmalayıcı
            def wrapper_fn(mode, img_path, text_input, background_path, query):
                try:
                    if mode == "Resim Yükle":
                        if not os.path.exists(background_path):
                            raise gr.Error("Arka plan dosyası bulunamadı!")
                        if not os.path.exists(img_path):
                            raise gr.Error("Nesne görseli bulunamadı!")
                        
                        background_image = Image.open(background_path).convert("RGB")
                        return all_function(img_path, query, background_image)
                    else:
                        # Metin girişi (URL) durumu
                        if not text_input.startswith(("http://", "https://")):
                            raise gr.Error("Geçerli bir URL giriniz!")
                            
                        # URL'den görseli indir ve işle
                        response = requests.get(text_input, timeout=10)
                        if response.status_code != 200:
                            raise gr.Error("URL'den görsel indirilemedi!")
                            
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        
                        if not os.path.exists(background_path):
                            raise gr.Error("Arka plan dosyası bulunamadı!")
                            
                        background_image = Image.open(background_path).convert("RGB")
                        return all_function(image, query, background_image)
                        
                except Exception as e:
                    raise gr.Error(f"İşlem sırasında hata oluştu: {str(e)}")

            submit_button.click(
                fn=wrapper_fn,
                inputs=[input_mode, input_img, input_text, background_img, query_text],
                outputs=output_image
            )



        with gr.TabItem("Görüntü-Metin Benzerliği (CLIP)"):
            gr.Markdown("**CLIP modeli ile görüntü ve metin arasındaki benzerlik analizi**")
            gr.Markdown("Aşağıya bir resim yükleyin ve virgülle ayırarak karşılaştırmak istediğiniz metinleri girin. Örneğin: `kedi, köpek, araba, deniz`")
            
            with gr.Row():
                image_input = gr.Image(type="pil", label="Görüntü Yükle")
                text_input = gr.Textbox(label="Metinleri girin (virgülle ayırın)", placeholder="örneğin: kedi, köpek, deniz")
            
            output = gr.Label(label="Benzerlik Sonuçları (Yüksekten düşüğe)")
            button = gr.Button("Benzerliği Hesapla")

            button.click(fn=clip_similarity_1, inputs=[image_input, text_input], outputs=output)
            
        with gr.TabItem("Görsel Soru-Cevap (Gemini)"): 
            gr.Markdown("**Gemini ile görsel tabanlı sohbet sistemi.**")
            gr.Markdown("Aşağıya bir resim yükleyin, sonra bu resimle ilgili bir soru sorun. Sohbet şeklinde devam edebilirsiniz.")

            chatbot = gr.Chatbot(label="Görsel Soru-Cevap Chatbot")
            image_input = gr.Image(type="pil", label="Görsel Yükle")
            question_input = gr.Textbox(label="Sorunuzu yazın ve Enter'a basın", placeholder="Resimde ne görüyorsun?")
            state = gr.State([])

            question_input.submit(
                fn=visual_question_answering_chatbot,
                inputs=[state, image_input, question_input],
                outputs=[chatbot, question_input]
            )
            
        with gr.TabItem("Görüntü Açıklama(BLIP)"): 
            gr.Markdown("**Image Captioning - Görüntüden metinsel açıklama üretme**")
            gr.Markdown("Bir resim yükleyin, model otomatik olarak açıklama üretsin.")

            image_input = gr.Image(type="pil", label="Görsel Yükle")
            caption_output = gr.Textbox(label="Görüntü Açıklaması", interactive=False)

            generate_button = gr.Button("Açıklama Oluştur")

            generate_button.click(
                fn=image_description,
                inputs=image_input,
                outputs=caption_output
            )
            
        with gr.TabItem("OCR Metin Tanıma(EasyOCR)"): 
            gr.Markdown("**Optical Character Recognition - Görüntüdeki metinleri okuma**")
            gr.Markdown("Bir resim yükleyin, model içindeki metni tanısın ve çıkarsın.")

            image_input = gr.Image(type="pil", label="Görsel Yükle")
            ocr_output = gr.Textbox(label="Tespit Edilen Metinler", lines=5, interactive=False)

            recognize_button = gr.Button("Metni Tanı")

            recognize_button.click(
                fn=extract_text_from_image,
                inputs=image_input,
                outputs=ocr_output
            )
            
        with gr.TabItem("Video Analizi"):
            gr.Markdown("**Video dosyalarından frame çıkarma ve analiz etme**")
            gr.Markdown("Bir video yükleyin. Sistem her 3 saniyede bir frame çıkarır, ardından CLIP ile en kaliteli 5 görseli seçer.")

            video_input = gr.Video(label="🎥 Video Yükle", height=300, width=400)

            top_5_images = gr.Gallery(label="🔝 En İyi 5 Frame", columns=5, height="auto", object_fit="contain")

            scores_textbox = gr.Textbox(label="🎯 Frame Skorları", lines=5, interactive=False)

            analyze_button = gr.Button("Analiz Et")

            analyze_button.click(
                fn=analyze_video,
                inputs=video_input,
                outputs=[top_5_images, scores_textbox]
            )
                    
        with gr.TabItem("Ses-Metin Dönüşümü (Whisper)"):
            gr.Markdown("**Whisper modeli ile konuşmayı metne çevirme**")
            
            audio_input = gr.Audio(label="🎧 Ses Dosyası Yükle", type="filepath")
            
            lang_dropdown = gr.Dropdown(
                label="Dil Seçin",
                choices=["en", "tr", "de", "fr", "es", "it", "ru"],
                value="en"
            )

            transcribe_button = gr.Button("Dönüştür")

            transcription_output = gr.Textbox(label="📝 Çıktı Metin", lines=6)

            transcribe_button.click(
                fn=transcribe_audio,
                inputs=[audio_input, lang_dropdown],
                outputs=transcription_output
            )
            
        with gr.TabItem("Müzik Analizi"):
            gr.Markdown("**🎵 Müzik dosyalarını analiz et ve türünü tahmin et**")

            music_file = gr.Audio(label="🎼 Müzik Dosyasını Yükle (.mp3 veya .wav)", type="filepath")
            analyze_button = gr.Button("🔍 Türü Tahmin Et")
            
            status = gr.Markdown("")  

            def genre_wrapper(file_path):
                status_message = "⏳ Tahmin yapılıyor, lütfen bekleyin..."
                yield status_message  

                result = classify_music_genre(file_path)
                yield result  

            analyze_button.click(
                fn=genre_wrapper,
                inputs=music_file,
                outputs=status
            )
            
        with gr.TabItem("Görsel Oluşturma (Stable Diffusion)"):
            gr.Markdown("**🖼️ Stable Diffusion ile metinden görsel oluşturma**")

            prompt_input = gr.Textbox(label="🎯 Prompt (Metin Girdisi)", placeholder="Örn: A futuristic city under the sea")
            generate_button = gr.Button("🎨 Görseli Oluştur")

            status = gr.Markdown("")
            output_image = gr.Image(label="🖼️ Oluşturulan Görsel")

            def stable_diffusion_wrapper(prompt):
                yield "⏳ Görsel oluşturuluyor, lütfen bekleyin...", None
                try:
                    image = stable_diffusion(prompt)
                    yield "✅ Görsel başarıyla oluşturuldu.", image
                except Exception as e:
                    yield f"❌ Hata oluştu: {e}", None

            generate_button.click(
                fn=stable_diffusion_wrapper,
                inputs=prompt_input,
                outputs=[status, output_image]
            )
            
        with gr.TabItem("Style Transfer"):
            gr.Markdown("**Görüntülere style transfer teknikleri uygulama**")

            with gr.Row():
                content_img_input = gr.Image(label="🖼️ İçerik Görüntüsü (Content Image)", type="filepath")
                style_img_input = gr.Image(label="🎨 Stil Görüntüsü (Style Image)", type="filepath")

            output_image = gr.Image(label="✨ Stil Aktarılmış Görüntü")

            apply_button = gr.Button("🎯 Stil Transferini Uygula")

            apply_button.click(
                fn=apply_style_transfer,
                inputs=[content_img_input, style_img_input],
                outputs=output_image
            )


demo.launch()