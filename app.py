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
    page_title="非计划再手术风险预测",
    page_icon="⚕️",
    layout="wide"
)

# 变量定义映射
VAR_DEFINITIONS = {
    "SEX": {0: "女性", 1: "男性"},
    "ASA scores": {0: "ASA < 3", 1: "ASA ≥ 3"},
    "tumor location": {
        1: "轴外且小脑上",
        2: "轴内且小脑上",
        3: "轴外且小脑下",
        4: "轴内且小脑下"
    },
    "Benign or malignant": {0: "良性", 1: "恶性"},
    "Admitted to NICU": {0: "未入NICU", 1: "入NICU"},
    "Duration of surgery": {0: "≤4小时", 1: ">4小时"},
    "diabetes": {0: "无糖尿病", 1: "有糖尿病"},
    "CHF": {0: "无CHF", 1: "有CHF"},
    "Functional dependencies": {0: "无功能依赖", 1: "有功能依赖"},
    "mFI-5": {
        0: "健康 (mFI = 0)",
        1: "预衰弱 (mFI = 1)",
        2: "衰弱 (mFI = 2)",
        3: "严重衰弱 (mFI ≥ 3)",
        4: "严重衰弱 (mFI ≥ 3)"
    },
    "Type of tumor": {
        1: "脑膜瘤",
        2: "原发性恶性脑肿瘤",
        3: "脑转移瘤",
        4: "听神经瘤",
        5: "其他"
    }
}

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
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
    st.title("⚕️ 非计划再手术风险预测系统")
    st.markdown("本系统基于机器学习算法，通过患者术前特征预测非计划再手术风险。")
    
    df = load_data()
    model, feature_names = train_model(df)
    
    # 侧边栏 - 变量详细定义
    with st.sidebar.expander("📚 变量定义", expanded=False):
        st.markdown("### 分类变量定义")
        for feature, values in VAR_DEFINITIONS.items():
            st.markdown(f"**{feature}**")
            for code, desc in values.items():
                st.markdown(f"- `{code}` = {desc}")
            st.markdown("")
    
    # 患者信息表单
    st.subheader("🔍 患者风险预测")
    with st.form("prediction_form"):
        # 使用三列布局显示输入字段
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(feature_names):
            col = cols[i % 3]  # 循环使用三列
            
            # 获取变量定义和默认值
            feature_def = VAR_DEFINITIONS.get(feature, {})
            default_val = int(df[feature].mean())
            
            with col:
                # 根据变量类型选择合适的输入组件
                if feature in VAR_DEFINITIONS:
                    # 分类变量使用下拉选择框
                    options = list(feature_def.keys())
                    option_labels = [f"{code} - {feature_def[code]}" for code in options]
                    
                    # 查找默认值在选项中的索引
                    default_idx = options.index(default_val) if default_val in options else 0
                    
                    selected_code = st.selectbox(
                        label=feature,
                        options=options,
                        index=default_idx,
                        format_func=lambda x: feature_def.get(x, str(x)),
                        help=f"可选值: {', '.join([f'{k} ({v})' for k, v in feature_def.items()])}"
                    )
                    input_data[feature] = selected_code
                else:
                    # 连续变量使用数字输入
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    
                    input_data[feature] = st.number_input(
                        label=feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=1
                    )
        
        # 提交按钮（居中显示）
        with st.columns(3)[1]:  # 中间列
            submitted = st.form_submit_button(
                "▶️ 开始预测",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            # 进行预测
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # 显示预测结果
            st.subheader("📊 预测结果")
            if prediction == 1:
                st.error(f"⚠️ **预测结果：高风险**")
                st.warning(f"非计划再手术概率: {proba:.2%}")
            else:
                st.success(f"✅ **预测结果：低风险**")
                st.info(f"非计划再手术概率: {proba:.2%}")
            
            # 生成SHAP解释图
            try:
                st.subheader("🔍 预测解释")
                st.markdown("下面的SHAP力场图展示了各特征对预测结果的影响程度：")
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 处理二分类SHAP值
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    class_idx = 1  # 高风险类别
                    shap_value = shap_values[class_idx]
                    expected_value = explainer.expected_value[class_idx]
                else:
                    class_idx = 0
                    shap_value = shap_values
                    expected_value = explainer.expected_value
                
                # 生成SHAP力场图
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    expected_value,
                    shap_value[0],
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                
                # 保存图像用于下载
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=300)
                buf.seek(0)
                
                # 下载按钮
                st.download_button(
                    label="📥 下载SHAP解释图",
                    data=buf,
                    file_name="shap_explanation.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.warning(f"生成SHAP解释图失败: {e}")
                st.markdown("可以尝试更新SHAP库: `pip install shap --upgrade`")

if __name__ == "__main__":
    main()
