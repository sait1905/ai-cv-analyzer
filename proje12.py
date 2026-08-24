import streamlit as st
import PyPDF2
import pandas as pd
import json
import time
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GOOGLE_API_KEY = "BURAYA_GEMINI_API_KEY_GIRIN"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash')

def pdf_to_text(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        st.error(f"PDF okunurken hata: {e}")
    return text.replace('\n', ' ').strip()

def hesapla_cosine_similarity(is_tanimi, cv_metni):
    if not is_tanimi or not cv_metni:
        return 0.0
    metinler = [is_tanimi, cv_metni]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(metinler)
        benzerlik = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(benzerlik * 100, 2)
    except:
        return 0.0

def analyze_cv_with_gemini(is_tanimi, cv_metni):
    prompt = f"""
    Sen kıdemli bir İK uzmanısın. Şu iş tanımına: {is_tanimi}
    Şu CV'yi değerlendir: {cv_metni}

    Bana SADECE aşağıdaki formatta, başka hiçbir açıklama yapmadan geçerli bir JSON döndür:
    {{
      "guc_yonler": ["madde 1", "madde 2"],
      "eksikler": ["madde 1", "madde 2"],
      "ik_ozeti": "Adayın genel durumu hakkında kısa ve profesyonel bir özet.",
      "mulakat_sorulari": ["2 adet zorlayıcı teknik mülakat sorusu."]
    }}
    """
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        
        ilk_suslu = text_response.find('{')
        son_suslu = text_response.rfind('}') + 1
        
        if ilk_suslu != -1 and son_suslu != 0:
            text_response = text_response[ilk_suslu:son_suslu]
        else:
            text_response = '{"guc_yonler": ["Bağlantı Hatası"], "eksikler": ["Bağlantı Hatası"], "ik_ozeti": "API aşırı yüklenmeden dolayı eksik yanıt verdi.", "mulakat_sorulari": ["Bağlantı Hatası"]}'
            
        return text_response
    except Exception as e:
        return '{"guc_yonler": ["API Kotası Doldu veya Hız Sınırı Aşıldı"], "eksikler": ["API Kotası Doldu veya Hız Sınırı Aşıldı"], "ik_ozeti": "Çok fazla CV tarandığı için Google API anlık hız sınırına (Rate Limit) takıldı. Lütfen biraz bekleyip tekrar deneyin.", "mulakat_sorulari": ["Lütfen biraz bekleyip tekrar deneyin."]}'

def compare_candidates_with_gemini(is_tanimi, aday1_isim, aday1_metin, aday2_isim, aday2_metin):
    prompt = f"""
    Sen kıdemli bir İK Direktörü ve Teknik Değerlendiricisin.
    
    İŞ TANIMI:
    {is_tanimi}
    
    ADAY 1 ({aday1_isim}):
    {aday1_metin}
    
    ADAY 2 ({aday2_isim}):
    {aday2_metin}
    
    Bu iki adayı bu pozisyon için doğrudan birbiriyle kıyasla. Hangisinin hangi alanda diğerine göre daha avantajlı olduğunu, aralarındaki belirleyici farkları ve son işe alım tercihini belirt.
    
    Bana SADECE aşağıdaki formatta, başka hiçbir açıklama yapmadan geçerli bir JSON döndür:
    {{
      "aday_1_avantajlari": ["Aday 1'in Aday 2'ye göre daha üstün olduğu 2-3 somut nokta"],
      "aday_2_avantajlari": ["Aday 2'nin Aday 1'e göre daha üstün olduğu 2-3 somut nokta"],
      "ayirici_farklar": ["İki aday arasındaki en belirleyici teknik veya tecrübe farkı"],
      "onerilen_aday": "Seçilen adayın tam dosya adı ({aday1_isim} veya {aday2_isim})",
      "secim_gerekcesi": "Bu pozisyonun gereksinimlerine göre bu adayın neden tercih edilmesi gerektiğinin 2-3 cümlelik net gerekçesi."
    }}
    """
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        
        ilk_suslu = text_response.find('{')
        son_suslu = text_response.rfind('}') + 1
        
        if ilk_suslu != -1 and son_suslu != 0:
            text_response = text_response[ilk_suslu:son_suslu]
        else:
            text_response = '{"aday_1_avantajlari": ["Hata"], "aday_2_avantajlari": ["Hata"], "ayirici_farklar": ["Bağlantı hatası"], "onerilen_aday": "Belirlenemedi", "secim_gerekcesi": "API yanıt veremedi."}'
            
        return text_response
    except Exception as e:
        return '{"aday_1_avantajlari": ["API Hatası"], "aday_2_avantajlari": ["API Hatası"], "ayirici_farklar": ["API Kotası"], "onerilen_aday": "Belirlenemedi", "secim_gerekcesi": "Google API hız sınırına takıldı."}'

st.set_page_config(page_title="Yapay Zeka Destekli ATS", page_icon="🤖", layout="wide")

if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False
    st.session_state.df_sonuclar = None
    st.session_state.is_tanimi = ""
    st.session_state.ai_hafizasi = {}
    st.session_state.karsilastirma_hafizasi = {}

st.title("🤖 Yapay Zeka Destekli Hibrit ATS (Aday Takip Sistemi)")
st.markdown("### Matematiksel Filtreleme, Yetenek Matrisi ve LLM Analiz Modülü")

is_tanimi = st.text_area("📋 İş Tanımını Giriniz:", height=150, placeholder="Aranan özellikleri ve kullanılacak teknolojileri buraya yazın...")

aranacak_yetenekler = st.text_input("🔍 Kontrol Edilecek Kritik Yetenekler (Virgülle ayırın):", value="Python, SQL, React, Docker")

yuklenen_dosyalar = st.file_uploader("📄 Adayların CV'lerini (PDF) Yükleyin (Çoklu Yükleme Aktif)", type="pdf", accept_multiple_files=True)

if st.button("🚀 CV'leri Analiz Et ve Sırala"):
    if is_tanimi and yuklenen_dosyalar:
        st.info("Sistem CV'leri işliyor, NLP motoru ve Yetenek Matrisi devrede...")
        
        yetenek_listesi = [y.strip() for y in aranacak_yetenekler.split(",") if y.strip()]
        sonuclar = []
        
        for dosya in yuklenen_dosyalar:
            cv_metni = pdf_to_text(dosya)
            skor = hesapla_cosine_similarity(is_tanimi, cv_metni)
            veri = {"Aday Dosyası": dosya.name, "Uyum Skoru (%)": skor, "Metin": cv_metni}
            
            for yetenek in yetenek_listesi:
                veri[yetenek] = "✅ Var" if yetenek.lower() in cv_metni.lower() else "❌ Yok"
            
            sonuclar.append(veri)
        
        df_sonuclar = pd.DataFrame(sonuclar)
        df_sonuclar = df_sonuclar.sort_values(by="Uyum Skoru (%)", ascending=False).reset_index(drop=True)
        
        st.session_state.df_sonuclar = df_sonuclar
        st.session_state.is_tanimi = is_tanimi
        st.session_state.analiz_yapildi = True
        st.success("✅ Matematiksel TF-IDF analizi ve yetenek eşleştirmeleri tamamlandı!")
    else:
        st.error("Lütfen önce iş tanımını girin ve en az bir adet PDF CV yükleyin.")

if st.session_state.analiz_yapildi:
    df_sonuclar = st.session_state.df_sonuclar
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Dinamik Aday Filtreleme")
        
        min_skor = st.slider("Minimum Uyum Skoru Barajı (%)", min_value=0, max_value=100, value=0)
        ozel_filtreler = st.text_input("Zorunlu Kriterleri Yazın (Virgülle Ayırın):", placeholder="Örn: İngilizce, 2 yıl, Senior, AWS")
        
        filtreli_df = df_sonuclar[df_sonuclar["Uyum Skoru (%)"] >= min_skor].copy()
        
        if ozel_filtreler:
            kriter_listesi = [k.strip() for k in ozel_filtreler.split(",") if k.strip()]
            for kriter in kriter_listesi:
                filtreli_df = filtreli_df[filtreli_df["Metin"].str.contains(kriter, case=False, na=False)]
        
        filtreli_df.index = range(1, len(filtreli_df) + 1)
        
        st.info(f"🎯 Filtreler sonucunda havuzdaki aday sayısı: **{len(filtreli_df)}**")
        st.markdown("💡 *Filtreler sol taraftaki tabloyu ve alttaki yapay zeka seçimlerini otomatik olarak günceller.*")
        
    with col1:
        st.subheader("🏆 Filtrelenmiş Liderlik Tablosu & Yetenek Matrisi")
        
        if len(filtreli_df) > 0:
            tablo_df = filtreli_df.drop(columns=["Metin"])
            st.dataframe(tablo_df, use_container_width=True)
            csv_verisi = tablo_df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Bu Listeyi CSV Olarak İndir",
                data=csv_verisi,
                file_name="filtrelenmis_adaylar.csv",
                mime="text/csv"
            )
        else:
            st.warning("Seçtiğiniz filtrelere uyan hiçbir aday bulunamadı.")
        
    st.markdown("---")
    st.subheader("🧠 Filtreden Geçen İlk 5 Aday İçin Yapay Zeka Analizi Raporu")
    
    top_5 = filtreli_df.head(5)
    
    if len(top_5) == 0:
        st.info("Filtreden geçen aday olmadığı için analiz raporu oluşturulamadı.")
    else:
        for index, row in top_5.iterrows():
            aday_dosyasi = row['Aday Dosyası']
            
            with st.expander(f"🥇 Sıra {index}: {aday_dosyasi} (Matematiksel Uyum: %{row['Uyum Skoru (%)']})", expanded=True):
                
                if aday_dosyasi not in st.session_state.ai_hafizasi:
                    with st.spinner(f"Gemini bu adayı inceliyor..."):
                        time.sleep(20)
                        ai_rapor_json = analyze_cv_with_gemini(st.session_state.is_tanimi, row["Metin"])
                        st.session_state.ai_hafizasi[aday_dosyasi] = ai_rapor_json
                
                rapor_json_metni = st.session_state.ai_hafizasi[aday_dosyasi]
                
                try:
                    rapor = json.loads(rapor_json_metni)
                    
                    st.write("**💪 Güçlü Yönleri:**")
                    for madde in rapor.get("guc_yonler", []):
                        st.write(f"✅ {madde}")
                    
                    st.write("")
                    st.write("**⚠️ Geliştirilmesi Gerekenler:**")
                    for madde in rapor.get("eksikler", []):
                        st.write(f"🔸 {madde}")
                        
                    st.write("")
                    st.info(f"**📝 İK Uzmanına Özet:** {rapor.get('ik_ozeti', '')}")
                    
                    st.write("")
                    st.write("**🎯 Önerilen Mülakat Soruları:**")
                    for soru in rapor.get("mulakat_sorulari", []):
                        st.write(f"❓ {soru}")
                        
                except json.JSONDecodeError:
                    st.error("Google API anlık hız sınırına takıldı. Lütfen biraz bekleyip sayfayı yenileyin.")

    st.markdown("---")
    st.subheader("⚔️ Aday Karşılaştırma & İkilem Çözücü (Head-to-Head)")
    st.markdown("Skorları yakın olan veya kararsız kaldığınız iki adayı seçerek yapay zekanın doğrudan aralarındaki farkları kıyaslamasını sağlayın.")
    
    tum_adaylar = df_sonuclar["Aday Dosyası"].tolist()
    
    if len(tum_adaylar) >= 2:
        karsilastirma_col1, karsilastirma_col2 = st.columns(2)
        
        with karsilastirma_col1:
            secilen_aday_1 = st.selectbox("1. Adayı Seçin:", tum_adaylar, index=0)
        with karsilastirma_col2:
            secilen_aday_2 = st.selectbox("2. Adayı Seçin:", tum_adaylar, index=1 if len(tum_adaylar) > 1 else 0)
            
        if secilen_aday_1 == secilen_aday_2:
            st.warning("Lütfen karşılaştırmak için iki farklı aday seçin.")
        else:
            if st.button("🔍 Bu İki Adayı Birebir Kıyasla"):
                karsilastirma_anahtari = f"{secilen_aday_1}_vs_{secilen_aday_2}"
                
                aday1_metin = df_sonuclar[df_sonuclar["Aday Dosyası"] == secilen_aday_1]["Metin"].values[0]
                aday2_metin = df_sonuclar[df_sonuclar["Aday Dosyası"] == secilen_aday_2]["Metin"].values[0]
                
                if karsilastirma_anahtari not in st.session_state.karsilastirma_hafizasi:
                    with st.spinner("Gemini iki adayı karşılıklı olarak inceliyor ve kıyaslama raporu hazırlıyor..."):
                        time.sleep(20)
                        karsilastirma_json = compare_candidates_with_gemini(
                            st.session_state.is_tanimi,
                            secilen_aday_1,
                            aday1_metin,
                            secilen_aday_2,
                            aday2_metin
                        )
                        st.session_state.karsilastirma_hafizasi[karsilastirma_anahtari] = karsilastirma_json
                
                sonuc_metni = st.session_state.karsilastirma_hafizasi[karsilastirma_anahtari]
                
                try:
                    karsilastirma_raporu = json.loads(sonuc_metni)
                    
                    st.success(f"🏆 **Yapay Zeka Tercihi:** {karsilastirma_raporu.get('onerilen_aday', '')}")
                    st.info(f"📌 **Seçim Gerekçesi:** {karsilastirma_raporu.get('secim_gerekcesi', '')}")
                    
                    st.write("**⚖️ Temel Ayrışma Noktaları:**")
                    for fark in karsilastirma_raporu.get("ayirici_farklar", []):
                        st.write(f"🔹 {fark}")
                        
                    st.write("")
                    
                    sol_karsilastirma, sag_karsilastirma = st.columns(2)
                    with sol_karsilastirma:
                        st.markdown(f"#### 👤 {secilen_aday_1} Üstün Yönleri")
                        for adv in karsilastirma_raporu.get("aday_1_avantajlari", []):
                            st.write(f"🟢 {adv}")
                            
                    with sag_karsilastirma:
                        st.markdown(f"#### 👤 {secilen_aday_2} Üstün Yönleri")
                        for adv in karsilastirma_raporu.get("aday_2_avantajlari", []):
                            st.write(f"🔵 {adv}")
                            
                except json.JSONDecodeError:
                    st.error("Karşılaştırma sonucu ayrıştırılırken hata oluştu veya API hız sınırına takıldı.")
    else:
        st.info("Aday karşılaştırması yapabilmek için sisteme en az 2 adet CV yüklenmiş olmalıdır.") 