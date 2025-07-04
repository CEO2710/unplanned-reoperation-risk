import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from io import BytesIO

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(
    page_title="Unplanned Reoperation Risk Prediction",
    page_icon="⚕️",
    layout="wide"
)

# 初始化会话状态（保存SHAP图的下载缓冲区）
if 'shap_plot_buf' not in st.session_state:
    st.session_state.shap_plot_buf = None

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# 训练模型
@st.cache_data
def train_model(df):
    X = df.drop("Unplanned reoperation", axis=1)
    y = df["Unplanned reoperation"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, X.columns

# 主应用
def main():
    st.title("❤️ Unplanned Reoperation Risk Prediction")
    st.markdown("This application uses machine learning to predict the risk of unplanned reoperation based on patient characteristics.")
    
    df = load_data()
    model, feature_names = train_model(df)
    
    # 侧边栏：变量定义
    with st.sidebar.expander("Variable Definitions", expanded=False):
        st.markdown("### Categorical Variable Definitions")
        st.markdown("""
        - **SEX**: 0 = Female, 1 = Male  
        - **ASA scores**: 0 = ASA < 3, 1 = ASA ≥ 3  
        - **tumor location**: 1=off-axis&subcerebellar, 2=intra-axis&subcerebellar, 3=off-axis&supracerebellar, 4=intra-axis&supracerebellar  
        - **Benign or malignant**: 0 = Benign, 1 = Malignant  
        - **Admitted to NICU**: 0 = No NICU, 1 = Admitted to NICU  
        - **Duration of surgery**: 0 = ≤4h, 1 = >4h  
        - **diabetes**: 0 = No diabetes, 1 = Diabetes  
        - **CHF**: 0 = No CHF, 1 = CHF  
        - **Functional dependencies**: 0 = No, 1 = Yes  
        - **mFI-5**: 0=Robust, 1=Pre-frail, 2=Frail, ≥3=Severely frail  
        - **Type of tumor**: 1=Meningiomas, 2=Primary malignant, 3=Metastatic, 4=Acoustic neuroma, 5=Other  
        """)
    
    # 用户预测表单
    st.subheader("🔍 Patient Risk Prediction")
    with st.form("prediction_form"):
        input_data = {}
        for feature in feature_names:
            min_val, max_val, mean_val = int(df[feature].min()), int(df[feature].max()), int(df[feature].mean())
            help_text = {
                "SEX": "0 = Female, 1 = Male",
                "ASA scores": "0 = ASA < 3, 1 = ASA ≥ 3",
                "tumor location": "1=off-axis&subcerebellar, 2=intra-axis&subcerebellar, 3=off-axis&supracerebellar, 4=intra-axis&supracerebellar",
                "Benign or malignant": "0 = Benign, 1 = Malignant",
                "Admitted to NICU": "0 = No NICU, 1 = Admitted to NICU",
                "Duration of surgery": "0 = ≤4h, 1 = >4h",
                "diabetes": "0 = No diabetes, 1 = Diabetes",
                "CHF": "0 = No CHF, 1 = CHF",
                "Functional dependencies": "0 = No, 1 = Yes",
                "mFI-5": "0=Robust, 1=Pre-frail, 2=Frail, ≥3=Severely frail",
                "Type of tumor": "1=Meningiomas, 2=Primary malignant, 3=Metastatic, 4=Acoustic neuroma, 5=Other"
            }.get(feature, None)
            
            input_data[feature] = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=1,
                help=help_text
            )
        
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # 显示预测结果
            st.subheader("📊 Prediction Result")
            if prediction == 1:
                st.error(f"Prediction: **High Risk of Unplanned Reoperation**")
                st.warning(f"Risk Probability: {proba:.2%}")
            else:
                st.success(f"Prediction: **Low Risk of Unplanned Reoperation**")
                st.info(f"Risk Probability: {proba:.2%}")
            
            # 生成并保存SHAP力图（不显示，直接保存）
            try:
                st.subheader("📥 Generate SHAP Force Plot for Download")
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 处理二分类SHAP值的索引问题
                if isinstance(shap_values, list):
                    class_idx = 1 if len(shap_values) > 1 else 0
                    expected_value = explainer.expected_value[class_idx]
                    shap_value = shap_values[class_idx]
                else:
                    class_idx = 0
                    expected_value = explainer.expected_value
                    shap_value = shap_values
                
                # 生成SHAP力图并保存到内存
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    expected_value,
                    shap_value[0],  # 单个样本的SHAP值
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                
                # 保存图像到会话状态
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=300)  # 提高DPI以获得更清晰的图像
                buf.seek(0)
                st.session_state.shap_plot_buf = buf
                
                plt.close(fig)  # 关闭图形以释放内存
                
                st.success("SHAP force plot generated successfully! Click below to download.")
                
            except Exception as e:
                st.error(f"Failed to generate SHAP plot: {str(e)}")
                st.write("Please check your SHAP version or input data format.")
    
    # 下载按钮（放在表单外）
    if st.session_state.shap_plot_buf is not None:
        st.download_button(
            label="📥 Download SHAP Force Plot (PNG)",
            data=st.session_state.shap_plot_buf,
            file_name="unplanned_reoperation_shap_plot.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
