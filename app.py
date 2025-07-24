import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# === Load model dan komponen ===
model = joblib.load('LogisticRegression - Yousician Learn Guitar & Bass.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Yousician Learn Guitar & Bass.pkl')
label_encoder = joblib.load('label_encoder_Yousician Learn Guitar & Bass.pkl')

# === Mapping label dan warna ===
label_map = {'positive': 'Positif', 'negative': 'Negatif'}
color_map = {'Positif': 'blue', 'Negatif': 'red'}

# === Judul Aplikasi ===
st.title("🎸 Aplikasi Analisis Sentimen – Yousician: Learn Guitar & Bass")

# === Pilih Mode Input ===
st.header("📌 Pilih Metode Input")
input_mode = st.radio("Pilih salah satu:", ["📝 Input Manual", "📁 Upload File CSV"])

# === Zona waktu lokal (WIB) ===
wib = pytz.timezone("Asia/Jakarta")
now_wib = datetime.now(wib)

# ========================================
# MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("🧾 Masukkan Satu Review Pengguna")

    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Rating Bintang:", [1, 2, 3, 4, 5])
    user_review = st.text_area("💬 Tulis Review Pengguna:")

    review_day = st.date_input("📅 Tanggal:", value=now_wib.date())
    review_time = st.time_input("⏰ Waktu:", value=now_wib.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_date_str = wib.localize(review_datetime).strftime("%Y-%m-%d %H:%M")

    if st.button("🚀 Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("⚠️ Silakan isi review terlebih dahulu.")
        else:
            vec = vectorizer.transform([user_review])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]

            result_df = pd.DataFrame([{
                "name": name if name else "(Anonim)",
                "star_rating": star_rating,
                "date": review_date_str,
                "review": user_review,
                "predicted_sentiment": label
            }])

            st.success(f"✅ Sentimen terdeteksi: **{label_map[label]}**")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                label="📥 Unduh Hasil sebagai CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name="hasil_prediksi_manual_Yousician_Guitar_Bass.csv",
                mime="text/csv"
            )

# ========================================
# MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("📄 Unggah File CSV Review")
    uploaded_file = st.file_uploader(
        "Pilih file CSV (dengan kolom: 'name', 'star_rating', 'date', 'review')",
        type=['csv']
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            required_cols = {'name', 'star_rating', 'date', 'review'}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ File harus memiliki kolom: {', '.join(required_cols)}.")
            else:
                df['review'] = df['review'].fillna("")
                X_vec = vectorizer.transform(df['review'])
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")

                # === Filter Tanggal ===
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()

                st.subheader("🗓️ Filter Rentang Tanggal")
                start_date = st.date_input("Mulai", min_value=min_date, max_value=max_date, value=min_date)
                end_date = st.date_input("Selesai", min_value=min_date, max_value=max_date, value=max_date)

                filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

                # === Filter Sentimen ===
                sentiment_option = st.selectbox("🎯 Filter Sentimen:", ["Semua", "Positif", "Negatif"])
                if sentiment_option == "Positif":
                    filtered_df = filtered_df[filtered_df['predicted_sentiment'] == "positive"]
                elif sentiment_option == "Negatif":
                    filtered_df = filtered_df[filtered_df['predicted_sentiment'] == "negative"]

                # === Tampilkan Tabel ===
                st.dataframe(
                    filtered_df[['name', 'star_rating', 'date', 'review', 'predicted_sentiment']],
                    use_container_width=True
                )

                # === Bar Chart ===
                st.subheader("📊 Distribusi Sentimen – Diagram Batang")
                sentimen_bahasa = filtered_df['predicted_sentiment'].map(label_map)
                bar_data = sentimen_bahasa.value_counts().reset_index()
                bar_data.columns = ['Sentimen', 'Jumlah']
                colors = [color_map.get(sent, 'gray') for sent in bar_data['Sentimen']]

                fig_bar, ax_bar = plt.subplots()
                bars = ax_bar.bar(bar_data['Sentimen'], bar_data['Jumlah'], color=colors)

                for bar in bars:
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                f"{int(height)}", ha='center', va='bottom')
                ax_bar.set_ylabel("Jumlah")
                ax_bar.set_xlabel("Sentimen")
                ax_bar.set_title("Distribusi Sentimen – Yousician: Learn Guitar & Bass")
                st.pyplot(fig_bar)

                # === Pie Chart ===
                st.subheader("🥧 Distribusi Sentimen – Diagram Pai")
                pie_data = sentimen_bahasa.value_counts()
                pie_colors = [color_map.get(sent, 'gray') for sent in pie_data.index]

                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(
                    pie_data,
                    labels=pie_data.index,
                    colors=pie_colors,
                    autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(pie_data)))})",
                    startangle=90
                )
                ax_pie.axis('equal')
                st.pyplot(fig_pie)

                # === Classification Report (jika ada label asli) ===
                if 'true_sentiment' in df.columns:
                    st.subheader("📈 Evaluasi Model (Jika Ada Label Asli)")
                    y_true = label_encoder.transform(df['true_sentiment'])
                    y_pred_enc = label_encoder.transform(df['predicted_sentiment'])

                    cm = confusion_matrix(y_true, y_pred_enc)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                                xticklabels=label_encoder.classes_,
                                yticklabels=label_encoder.classes_, ax=ax_cm)
                    ax_cm.set_xlabel("Prediksi")
                    ax_cm.set_ylabel("Label Asli")
                    ax_cm.set_title("Confusion Matrix")
                    st.pyplot(fig_cm)

                    report = classification_report(y_true, y_pred_enc, target_names=label_encoder.classes_)
                    st.text("Classification Report:")
                    st.text(report)

                # === Unduh CSV ===
                csv_result = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil CSV",
                    data=csv_result,
                    file_name="hasil_prediksi_Yousician_Guitar_Bass.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
