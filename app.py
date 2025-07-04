import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="Unplanned Reoperation Risk Prediction",
    page_icon="⚕️",
    layout="wide"
)

# 变量定义（仅内部使用，不显示在界面）
VAR_DEFINITIONS = {
    "SEX": {
        "Female": "Female",
        "Male": "Male"
    },
    "ASA scores": {0: "ASA < 3", 1: "ASA ≥ 3"},
    "tumor location": {
        1: "Off-axis AND Supracerebellar",
        2: "Intra-axis AND Supracerebellar",
        3: "Off-axis AND Subcerebellar",
        4: "Intra-axis AND Subcerebellar"
    },
    "Benign or malignant": {0: "Benign", 1: "Malignant"},
    "Admitted to NICU": {0: "No NICU", 1: "Admitted to NICU"},
    "Duration of surgery": {0: "≤4 hours", 1: ">4 hours"},
    "diabetes": {0: "No diabetes", 1: "Diabetes"},
    "CHF": {0: "No CHF", 1: "CHF"},
    "Functional dependencies": {0: "No", 1: "Yes"},
    "mFI-5": {  # 严格分为四类
        0: "Robust (mFI = 0)",
        1: "Pre-frail (mFI = 1)",
        2: "Frail (mFI = 2)",
        3: "Severely frail (mFI ≥ 3)"
    },
    "Type of tumor": {
        1: "Meningiomas",
        2: "Primary malignant brain tumors",
        3: "Metastatic brain tumor",
        4: "Acoustic neuroma",
        5: "Other"
    }
}

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        # 处理SEX列：将原数据中的0/1映射为Female/Male（如果需要）
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({0: "Female", 1: "Male"})
        return df
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

# 训练模型（处理SEX变量的映射）
@st.cache_data
def train_model(df):
    # 临时映射SEX回0/1用于模型训练
    if 'SEX' in df.columns:
        df = df.copy()
        df['SEX'] = df['SEX'].map({"Female": 0, "Male": 1})
    
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
    # 初始化会话状态
    if 'shap_buf' not in st.session_state:
        st.session_state.shap_buf = None
    
    st.title("⚕️ Unplanned Reoperation Risk Prediction")
    st.markdown("Predicts unplanned reoperation risk using machine learning.")
    
    df = load_data()
    model, feature_names = train_model(df)
    
    # 患者信息表单
    st.subheader("🔍 Patient Risk Assessment")
    with st.form("prediction_form"):
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                # 特殊处理SEX：直接显示Male/Female
                if feature == "SEX":
                    # 下拉框直接显示Male/Female
                    input_data[feature] = st.selectbox(
                        "Sex",
                        options=["Female", "Male"],
                        index=0  # 默认显示Female
                    )
                # 处理mFI-5（严格四类）
                elif feature == "mFI-5":
                    input_data[feature] = st.selectbox(
                        "mFI-5",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: VAR_DEFINITIONS["mFI-5"][x],
                        index=0
                    )
                # 其他分类变量
                elif feature in VAR_DEFINITIONS:
                    options = list(VAR_DEFINITIONS[feature].keys())
                    input_data[feature] = st.selectbox(
                        feature,
                        options=options,
                        format_func=lambda x: VAR_DEFINITIONS[feature][x]
                    )
        
        # 预测按钮
        submitted = st.form_submit_button("▶️ Predict Risk", type="primary")
        
        if submitted:
            # 处理SEX的映射（转为模型需要的0/1）
            input_df = pd.DataFrame([input_data])
            if 'SEX' in input_df.columns:
                input_df['SEX'] = input_df['SEX'].map({"Female": 0, "Male": 1})
            
            # 模型预测
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # 显示结果
            st.subheader("📊 Prediction Results")
            if prediction == 1:
                st.error("⚠️ **High Risk of Unplanned Reoperation**")
                st.warning(f"Probability: {proba:.2%}")
            else:
                st.success("✅ **Low Risk of Unplanned Reoperation**")
                st.info(f"Probability: {proba:.2%}")
            
            # 生成SHAP图
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 处理SHAP值
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_value = shap_values[1][0]
                    expected_value = explainer.expected_value[1]
                else:
                    shap_value = shap_values[0]
                    expected_value = explainer.expected_value
                
                # 绘制SHAP图
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                
                # 保存到会话状态
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=300)
                buf.seek(0)
                st.session_state.shap_buf = buf
                plt.close(fig)
                
                st.success("SHAP plot generated! Download option below.")
                
            except Exception as e:
                st.error(f"SHAP plot failed: {e}")
                st.markdown("Try `pip install shap --upgrade`")
    
    # 表单外的下载按钮
    if st.session_state.shap_buf is not None:
        st.download_button(
            "📥 Download SHAP Plot",
            data=st.session_state.shap_buf,
            file_name="shap_force_plot.png",
            mime="image/png"
        )
        st.session_state.shap_buf = None  # 清空缓存

if __name__ == "__main__":
    main()
