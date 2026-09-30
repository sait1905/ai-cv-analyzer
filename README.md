# AI-Powered ATS & Candidate Arbiter (Aday Takip Sistemi)

Bu proje; İnsan Kaynakları süreçleri için Bilgi Erişimi (Information Retrieval) ve Üretken Yapay Zeka (LLM) teknolojilerini bir araya getiren hibrit bir Aday Takip ve Karar Destek Sistemi'dir.

## 🚀 Temel Özellikler
- **Matematiksel Ön Eleme:** PyPDF2 ile CV metin ayıklama, TF-IDF vektörleştirme ve Kosinüs Benzerliği ile nesnel skorlama.
- **Yetenek Matrisi & Filtreleme:** Büyük/küçük harf duyarsız dinamik yetenek eşleştirme, minimum skor barajı ve çoklu kriter (AND) filtreleme.
- **Niteliksel LLM Analizi:** Google Gemini API ile filtrelenen en başarılı adayların güçlü/zayıf yön analizi ve pozisyona özel teknik mülakat soruları.
- **Birebir Karar Hakemi (Head-to-Head Arbiter):** Kararsız kalınan iki adayın yapay zeka hakemliğinde gerekçeli kıyaslanması.
- **Performans & Güvenlik:** Streamlit oturum durumu (`session_state`) ile API kota koruması, önbellekleme ve Türkçe karakter destekli (`utf-8-sig`) CSV dışa aktarımı.

## 🛠️ Teknolojiler
- Python
- Streamlit
- Scikit-learn
- Google Generative AI (Gemini API)
- PyPDF2 & Pandas

## 💻 Kurulum ve Çalıştırma
1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/sait1905/ai-cv-analyzer.git
   cd ai-cv-analyzer
   pip install -r requirements.txt
   py -m streamlit run proje12.py
   ```
   > **Not:** Projeyi çalıştırmadan önce `proje12.py` dosyasındaki `GOOGLE_API_KEY` alanına kendi Google Gemini API anahtarınızı tanımlayın.
