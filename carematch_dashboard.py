import streamlit as st

st.markdown(""" ***GROUP 4***: TU PHAM & MINH NGUYEN""")
# === Dashboard Title ===
st.title("📊 Carematch Dashboard")

# === Load Dataset ===
import pandas as pd
carematch = pd.read_csv("carematch_requests.csv")
# === Introduction / Project Background ===
st.header("🏥 Project Background")
st.markdown("""**CareMatch Health** is a regional healthcare network serving a diverse patient population across both urban and suburban communities.
Patients submit appointment requests and complete intake forms through the organization’s digital platforms.

Although CareMatch holds a large volume of patient and operational data, it has not yet implemented advanced analytics or AI-powered tools to derive value from this information.
➡️ As a result, the immediate need is to **explore the data, identify opportunities, extract actionable insights, and build data-driven solutions** that can improve access, efficiency, and patient experience.
""")
# === Show Sample Data ===
st.subheader("Sample Data")
st.write(carematch.head())

# === Descriptive Stats ===
st.header("📊 Descriptive Statistics (All Variables)")
desc_stats = carematch.describe(include="all").T
st.dataframe(desc_stats)
# === Histogram Plots ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Wait Time Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(carematch['wait_time'], bins=20, kde=False, color='blue', ax=ax1)
    st.pyplot(fig1)
st.markdown(""" Wait time are spread out without a strong concentration at a particular interval""")
with col2:
    st.subheader("Chronic Conditions Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(carematch['chronic_conditions_count'], bins=20, kde=False, color='blue', ax=ax2)
    st.pyplot(fig2)
st.markdown(""" Most patients present with 0–2 chronic conditions, with 1 chronic condition being the most common.
This distribution highlights that while the majority of cases are relatively simple, resource planning should account for a smaller group of patients with complex healthcare needs.""")
# === Boxplots ===
st.header("📊 Wait Time by Categories")
# --- Conclusion for Wait Time Analysis ---
st.markdown("""
### ✅ Conclusion: Wait Time Analysis

- Wait times are fairly consistent across **language preference, provider specialty, and urgency score**.
- The **median wait time is ~15 days** for all groups, with wide variability.
- This suggests that **individual patient characteristics and provider type do not strongly impact wait times**.
- Instead, delays may be driven more by **system-level factors** such as scheduling efficiency and resource allocation.
- ⚠️ Notably, **urgency score does not significantly reduce wait times**, highlighting a **misalignment between clinical need and scheduling practices**.
""")
st.subheader("Wait Time by Language Preference")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="language_pref", y="wait_time", palette="Set3", ax=ax3)
st.pyplot(fig3)

st.subheader("Wait Time by Provider Specialty")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="provider_specialty", y="wait_time", palette="Set3", ax=ax4)
st.pyplot(fig4)

st.subheader("Wait Time by Urgency Score")
fig5, ax5 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="urgency_score", y="wait_time", palette="Set3", ax=ax5)
st.pyplot(fig5)

# === Countplots ===
st.header("📊 Distribution of Categorical Variables")
col3, col4 = st.columns(2)
st.markdown("""**Urgency Score Distribution** is fairly balanced across all five levels, indicating that patients are being assigned urgency ratings in a relatively even manner.
**Mental Health Flag** shows a strong imbalance: the vast majority of requests (~85%) are **not flagged for mental health**, while only a small fraction (~15%) are.""")
with col3:
    st.subheader("Urgency Score Distribution")
    fig6, ax6 = plt.subplots(figsize=(8,5))
    sns.countplot(data=carematch, x="urgency_score", order=carematch['urgency_score'].value_counts().index, ax=ax6)
    st.pyplot(fig6)

with col4:
    st.subheader("Mental Health Flag Distribution")
    fig7, ax7 = plt.subplots(figsize=(8,5))
    sns.countplot(data=carematch, x="mental_health_flag", order=carematch['mental_health_flag'].value_counts().index, ax=ax7)
    st.pyplot(fig7)

# === Word Cloud ===
st.header("☁️ Word Cloud of Condition Summaries")
st.markdown("""The word cloud provides a **quick thematic snapshot** of what patients are most frequently seeking help for, guiding providers on where to focus resources.""")
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

carematch['clean_summary'] = carematch['condition_summary'].apply(preprocess)
text = " ".join(carematch['clean_summary'])
stopwords = set(STOPWORDS)
stopwords.update(["need","ongoing","consultation","requesting","follow","patient"])

wordcloud = WordCloud(width=1200, height=600, background_color="white",
                      stopwords=stopwords, colormap="tab10", collocations=True).generate(text)

fig8, ax8 = plt.subplots(figsize=(12,6))
ax8.imshow(wordcloud, interpolation="bilinear")
ax8.axis("off")
st.pyplot(fig8)
# === Case & Provider Counts with Filters ===
st.header("📊 Case & Provider Counts with Filters")
st.sidebar.header("🔎 Filters")
st.markdown("""
- ***Provider Coverage by Location:** How many unique providers are available within each zip code?

- ***Workload Distribution by Month:*** How many patient cases are assigned to each provider on a monthly basis?

- ***Provider Case Volume:*** How many total cases each provider ID is responsible for managing, reflecting workload intensity.""")

zip_options = sorted(carematch['zip_code'].dropna().unique())
provider_options = sorted(carematch['assigned_provider_id'].dropna().unique())
selected_zip = st.sidebar.selectbox("Select a Zip Code", ["All"] + list(zip_options))
selected_provider = st.sidebar.selectbox("Select a Provider ID", ["All"] + list(provider_options))

# Cases per zip
cases_per_zip = carematch['zip_code'].value_counts().reset_index()
cases_per_zip.columns = ['zip_code', 'total_cases']
providers_per_zip = carematch.groupby("zip_code")["assigned_provider_id"].nunique().reset_index(name="unique_providers")
zip_summary = pd.merge(cases_per_zip, providers_per_zip, on="zip_code")
if selected_zip != "All":
    zip_summary = zip_summary[zip_summary['zip_code'] == selected_zip]
st.subheader("📍 Zip Code Summary")
st.dataframe(zip_summary)

# Provider case counts
provider_case_counts = carematch['assigned_provider_id'].value_counts().reset_index()
provider_case_counts.columns = ['assigned_provider_id', 'total_cases_for_provider']
if selected_provider != "All":
    provider_case_counts = provider_case_counts[provider_case_counts['assigned_provider_id'] == selected_provider]
st.subheader("👨‍⚕️ Provider Case Counts")
st.dataframe(provider_case_counts)

# Cases per provider within zip
zip_provider_cases = carematch.groupby(["zip_code", "assigned_provider_id"]).size().reset_index(name="case_count")
if selected_zip != "All":
    zip_provider_cases = zip_provider_cases[zip_provider_cases['zip_code'] == selected_zip]
if selected_provider != "All":
    zip_provider_cases = zip_provider_cases[zip_provider_cases['assigned_provider_id'] == selected_provider]
st.subheader("📍+👨‍⚕️ Cases per Provider within each Zip Code")
st.dataframe(zip_provider_cases)

# === Monthly Case Counts ===
st.header("📅 Monthly Case Counts per Provider")
carematch['request_timestamp'] = pd.to_datetime(carematch['request_timestamp'])
carematch['request_month'] = carematch['request_timestamp'].dt.to_period("M")

monthly_counts = carematch.groupby(['assigned_provider_id','request_month']).size().reset_index(name='case_count')
years = sorted(carematch['request_timestamp'].dt.year.unique())
months = sorted(carematch['request_timestamp'].dt.month.unique())
selected_year = st.sidebar.selectbox("Select Year", ["All"] + list(years))
selected_month = st.sidebar.selectbox("Select Month", ["All"] + list(months))

filtered = monthly_counts.copy()
if selected_year != "All":
    filtered = filtered[filtered['request_month'].dt.year == int(selected_year)]
if selected_month != "All":
    filtered = filtered[filtered['request_month'].dt.month == int(selected_month)]
st.subheader("📊 Case Counts per Provider (Filtered by Month/Year)")
st.dataframe(filtered)
# === Keyword Extraction ===
st.markdown("""***Data Preprocessing***:
The dataset contained free-text entries under the column **condition_summary**.
We extract a concise diagnosis keyword from each summary using **YAKE** to standardize inputs for clustering.""")

st.header("🩺 Keyword Extraction from Condition Summaries")

kw_extractor = yake.KeywordExtractor(top=1, stopwords=None)

def extract_keyword(text):
    if pd.isnull(text) or not str(text).strip():
        return None
    keywords = kw_extractor.extract_keywords(str(text))
    return keywords[0][0] if keywords else None

if "diagnosis" not in carematch.columns:
    carematch["diagnosis"] = carematch["condition_summary"].apply(extract_keyword)

keyword_counts = (carematch['diagnosis']
                  .dropna()
                  .value_counts()
                  .reset_index())
keyword_counts.columns = ["diagnosis_keyword","count"]

st.subheader("Most Frequent Diagnosis Keywords")
st.dataframe(keyword_counts.head(20))

# Bar plot of top keywords
fig9, ax9 = plt.subplots(figsize=(10,6))
sns.barplot(data=keyword_counts.head(15), x="count", y="diagnosis_keyword", ax=ax9)
ax9.set_xlabel("Count"); ax9.set_ylabel("Diagnosis keyword")
st.pyplot(fig9)
# Count diagnosis keyword frequency by provider_specialty
keyword_by_specialty = (
    carematch.dropna(subset=["diagnosis", "provider_specialty"])
            .groupby(["provider_specialty", "diagnosis"])
            .size()
            .reset_index(name="count")
)

# Get top 5 keywords per specialty
top_keywords_per_specialty = (
    keyword_by_specialty
    .sort_values(["provider_specialty", "count"], ascending=[True, False])
    .groupby("provider_specialty")
    .head(5)
)
print(top_keywords_per_specialty.head(20))
st.subheader("🔑 Top Diagnosis Keywords by Provider Specialty")

pivot = keyword_by_specialty.pivot_table(
    index="diagnosis",
    columns="provider_specialty",
    values="count",
    aggfunc="sum",
    fill_value=0
)

plt.figure(figsize=(12,8))
sns.heatmap(pivot.head(20), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Top Keywords by Provider Specialty")
st.pyplot(plt)

# === Clustering  ====
# ====================
st.header("🤖 Patient Clustering Analysis")
st.markdown("""***Method:*** We combine TF-IDF vectors of the extracted **diagnosis** with three structured signals (**urgency_score**, **chronic_conditions_count**, **mental_health_flag**).
We then run **MiniBatchKMeans** (works with sparse matrices) and visualize clusters with **TruncatedSVD** (PCA-like for sparse).""")

# Keep only rows that have a diagnosis keyword
mask = carematch["diagnosis"].notnull()
if mask.sum() < 5:
    st.warning("Not enough rows with extracted diagnosis to run clustering (need at least 5).")
else:
    try:
        # ---- Vectorize diagnosis (sparse) ----
        vectorizer = TfidfVectorizer(stop_words="english")
        X_text = vectorizer.fit_transform(carematch.loc[mask, "diagnosis"].astype(str))

        # ---- Scale structured features ----
        cluster_scaler = StandardScaler()
        X_struct = cluster_scaler.fit_transform(
            carematch.loc[mask, ["age", "urgency_score", "chronic_conditions_count", "mental_health_flag"]]
        )

        # ---- Fuse into one sparse-like design ----
        X_cluster = hstack([X_text, X_struct])

        # ---- Elbow method ----
        st.header("📉 Elbow Method for Optimal k")
        inertia = []
        K = range(2, 11)
        for k_opt in K:
            kmb = MiniBatchKMeans(n_clusters=k_opt, random_state=42, n_init=10, batch_size=2048)
            kmb.fit(X_cluster)
            inertia.append(kmb.inertia_)
        fig10, ax10 = plt.subplots(figsize=(8,6))
        ax10.plot(list(K), inertia, "bo-")
        ax10.set_xlabel("Number of clusters (k)")
        ax10.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
        st.pyplot(fig10)

        # ---- Sidebar: choose k ----
        st.sidebar.subheader("⚙️ Clustering Parameters")
        k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=2048)
        labels = kmeans.fit_predict(X_cluster)
        carematch.loc[mask, "cluster"] = labels

        # ---- 2D visualization ----
        st.subheader("📊 2D Visualization of Clusters")
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_2d = svd.fit_transform(X_text)
        fig11, ax11 = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            x=X_2d[:,0], y=X_2d[:,1],
            hue=carematch.loc[mask, "cluster"].astype(int),
            palette="tab10", ax=ax11, legend=True
        )
        ax11.set_xlabel("Component 1"); ax11.set_ylabel("Component 2")
        st.pyplot(fig11)

        # ---- Cluster insights ----
        st.subheader("📑 Cluster Insights")
        st.markdown("""Patients with similar diagnosis keywords are grouped together.
        Structured features help separate acute vs. chronic/long-term management groups.""")

        for c in sorted(carematch.loc[mask, "cluster"].unique()):
            subset = carematch.loc[carematch["cluster"] == c]
            st.markdown(f"### 🔹 Cluster {int(c)} Summary")

            # Top 5 diagnosis keywords
            top_diag = subset["diagnosis"].value_counts().head(5)
            top_diag_df = top_diag.reset_index()
            top_diag_df.columns = ["diagnosis", "count"]
            st.dataframe(top_diag_df)

            # Numeric summaries
            st.write("**Avg Age:**", round(subset["age"].mean(), 1))
            st.write("**Avg Urgency:**", round(subset["urgency_score"].mean(), 2))
            st.write("**Avg Chronic Conditions:**", round(subset["chronic_conditions_count"].mean(), 2))
            st.write("**Mental Health Flag %:**", round(subset["mental_health_flag"].mean()*100, 2), "%")

        # ---- Wait Time ----
        st.subheader("⏱️ Wait Time Distribution by Cluster")
        if "wait_time" in carematch.columns:
            fig12, ax12 = plt.subplots(figsize=(8,6))
            sns.boxplot(x="cluster", y="wait_time", data=carematch.loc[mask], ax=ax12)
            st.pyplot(fig12)

        # ---- Provider Specialty ----
        st.subheader("🏥 Provider Specialty Distribution by Cluster")
        if "provider_specialty" in carematch.columns:
            fig13, ax13 = plt.subplots(figsize=(12,6))
            sns.countplot(x="cluster", hue="provider_specialty", data=carematch.loc[mask], ax=ax13)
            ax13.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            st.pyplot(fig13)

        # ---- Conclusion ----
        st.subheader("📑 ***CLUSTER CONCLUSION***")
        st.markdown("Use these clusters as priors inside the Triage Assistant to guide specialty routing and expected wait times.")

    except Exception as e:
        st.exception(e)
st.markdown("""***Key Takeaways***

- Clusters are not distinguished by wait time, but by provider specialty demand.

- Resource allocation should therefore focus on specialty coverage rather than purely reducing wait times.

- Cluster 1 and Cluster 3 represent the highest patient loads and may require more staffing and scheduling flexibility to balance demand.

- Clusters 0 and 2, though smaller, should not be overlooked as they might represent unique patient needs (e.g., targeted chronic conditions or specific demographics).""")

st.markdown("""***CONCLUSION***
- Our goal of the project is to improve wait time for patients’ appointment through analyzing the symptoms and the information about the patient such as zip code, provider specialty, age.
  However, our analysis shows no meaningful wait time improvement even with clustering, suggesting that more information needed for dataset over a long period of time, thus the robustness of the dataset would yield more meaningful insights during the data analysis process.""")
