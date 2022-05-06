import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, plot_roc_curve, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import shap
shap.initjs()


# Load data
@st.cache
def load_data():
    df = pd.read_csv('data/HR_comma_sep.csv')
    return df

# load the model
model = pickle.load(open('utils/model.pkl', 'rb'))
oe = pickle.load(open('utils/oe_scaler.pkl', 'rb'))
le = pickle.load(open('utils/le_scaler.pkl', 'rb'))
ss = pickle.load(open('utils/ss_scaler.pkl', 'rb'))

def explorer():
    st.write("モデルとデータの説明")

    with st.expander("データの説明"):

            
        st.subheader("データの説明")


        '''
            従業員14,999名のデータ。以下の10の変数から構成。

            - satisfaction_level: 従業員の満足度（0-1）
            - last_evaluation: 従業員の上司の評価（0-1）
            - number_projects: 担当プロジェクトの件数
            - average_monthly_hours: １か月の平均労働時間
            - time_spent_company: 勤続年数
            - work_accident: 従業員による過去の事故発生有無（1=あり、0=なし）
            - promotion_last_5years: 過去５年の昇進の有無（1=あり、0=なし）
            - department: 所属部署
            - salary: 給与水準（low, medium, high)
            - left: 従業員の退職の有無（1=あり、0=なし）
        '''

        df = load_data()
        
        # display data
        st.subheader("サンプルデータ")
        st.dataframe(df, width = 1000)
        link_data = '[github](https://github.com/simmieyungie/HR-Analytics/tree/master)'
        st.markdown('出展：' + link_data, unsafe_allow_html=True)   

    with st.expander('モデルの説明'):
        st.subheader('モデルについて')
        '''
        モデルは「教師付き機械学習」を用いました。「教師付き機械学習」とは、正解データとなる目的変数（今回は従業員の離職の有無）に対して、複数の説明変数により、パターン分けを行い、正解データを予測するものです。  

        今回は、「離職の有無」の予測なので、従業員の離職があるか（目的変数=1）、ないか（目的変数=0）、という「教師付き機械学習」の「二値分類問題」となります。  
        
        例えば、従業員満足度（satisfaction_level）が極めて高いが、給与水準（salary）がとても低い場合などは、離職の可能性が高まるため、説明変数をパターン化することで予測精度を高めることができます。 
        '''
        
        st.subheader('アルゴリズムについて')
        '''
        説明変数のパターン化に用いるのがアルゴリズムですが、今回は予測精度が高く、かつ学習速度の早い「LightGBMモデル」を用いています。  

        LightGBMは2017年にMicrosoftのチームによって開発されたモデルで、決定木モデル（樹形図のように閾値の大小などでデータのパターンを分岐させるもの）を複数組み合わせ、学習制度を改善していくBoostingという方法を取っています。  

        イメージは以下の図のとおりとなり、最初の決定木から学習をすることで、精度をあげていくことができるモデルになります。
        '''
        st.image('https://www.researchgate.net/publication/350165006/figure/fig2/AS:1020116934344707@1620226238876/Leafwise-growth-of-LightGBM-classifier.png')
        link_image = '[researchgate](https://www.researchgate.net/publication/350165006/figure/fig2/AS:1020116934344707@1620226238876/Leafwise-growth-of-LightGBM-classifier.png)'
        st.markdown('出展：' + link_image, unsafe_allow_html=True)   

        st.subheader('精度について')
        '''
        二値分類の場合は、4つの精度を用いることが一般的です。（TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative）
        - Accuracy（正解率）：(TP + TN) / (TP + TN + FP + FN)。予測の正しさを見る指標。ただし、データが不均衡の場合は、正確性を欠く（例えば、目的変数の0が99個あり、1が1個の場合、モデルで0と予測しておけば、99％の正答率となってしまう）
        - Precision（適合率）：(TP) / (TP + FP)。正と予測したものが、どれだけ正しかったかを見る指標
        - Recall（再現率）：(TP) / (TP + FN)。実際に正であったもののうち、どれだけ正と予測できたかを見る指標
        - F1：(2Precision x Recall) / (Precision + Recall)。PrecisionとRecallの調和平均。Fが高いというのはPrecisionとRecallがバランス良く高い値となる。
        '''

    with st.expander('サンプルデータで予測してみる'):
        sample_data_button = st.checkbox('サンプルデータを取得')

        if sample_data_button:
            df = load_data()
            X_test_trans, y_test = sample_data_prep(df)

            sample_prediction_button = st.checkbox('サンプルデータから退職の有無を予測！')
            if sample_prediction_button:
                
                pred = model.predict(X_test_trans)
                pred_series = pd.DataFrame(pred, columns = ['退職の可能性'])
                pred_series['退職の可能性'] = ['低い' if elem == 0 else '高い' for elem in pred_series['退職の可能性']]

                pred_prob = pd.DataFrame(model.predict_proba(X_test_trans), columns = ['継続確率', '退職確率（pct）']).loc[:, '退職確率（pct）']
                pred_prob = round(pred_prob * 100)
                # pred_prob['退職確率（pct）'] = [round(elem * 100, 2) for elem in pred_prob['退職確率（pct）']]
               
                pred_res = pd.concat([pred_series, pred_prob], axis = 1)      

                st.subheader('予測結果')
                st.table(pred_res.head())

                # Show  scores
                precision_res = precision_score(y_test, pred)
                recall_res = recall_score(y_test, pred)
                accuracy_res = accuracy_score(y_test, pred)
                f1_res = f1_score(y_test, pred)

                st.subheader('モデルスコア')
                st.write('- Accuracy: '+ str(round(accuracy_res * 100, 2)))
                st.write('- Precision: '+ str(round(precision_res * 100, 2)))
                st.write('- Recall: '+ str(round(recall_res * 100, 2)))
                st.write('- F1: '+ str(round(f1_res * 100, 2)))

                fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout = True, figsize = (7, 3))
                ax1.set_title('AUC_ROC_Curve', fontsize = 16)
                plot_roc_curve(model, X_test_trans, y_test, ax = ax1)
                ax2.set_title('Confusion Matrix', fontsize = 16)
                plot_confusion_matrix(model, X_test_trans, y_test, cmap = 'Blues', ax = ax2)
                st.pyplot(fig)

                # shap
                col_list = list(df.columns)
                col_list.remove('left')

                X_shap = pd.DataFrame(X_test_trans, columns = col_list)
                st.write(list(df.columns).remove('left'))
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                fig_shap, ax = plt.subplots(figsize = (12, 6))
                shap.summary_plot(shap_values, X_shap, plot_type = 'bar', plot_size=None)
                st.subheader('モデルに対する各変数の寄与度の可視化')
                st.pyplot(fig_shap)


# Preprocess data
def sample_data_prep(df):
    df_ = df.sample(100).reset_index(drop = True)
    st.subheader('ランダムに取得したサンプルデータ\n(100件中5件を表示)')
    st.table(df_.head())
    df_['salary'] = oe.transform(df_[['salary']])
    df_['sales'] = le.transform(df_['sales'])
    X_test, y_test = df_.drop(columns = ['left']), df_['left']
    X_test_trans = ss.transform(X_test)
    return X_test_trans, y_test


def data_input():
    # form sample
    with st.form('prediction_form'):
        satifsaction_level = st.slider("現在の満足度（100点中）", min_value = 0, max_value = 100)
        last_evaluation = st.select_slider("直近の評価結果（10点満点）", options = range(11))
        number_project = st.select_slider("経験したプロジェクトの数", options = range(7))
        average_montly_hours = st.number_input('月平均労働時間（hour)', min_value = 0, max_value = 372) # 12 hour x 31 days
        time_spend_company = st.select_slider('入社してからの年数', options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        Work_accident = st.radio('事故経験', options = ['あり', 'なし'])
        promotion_last_5years = st.radio('過去５年間の昇進', options = ['あり', 'なし'])
        sales = st.selectbox('所属部門', options = ('sales', 'accounting', 'hr', 'technical', 'support', 'management',
                                                    'IT', 'product_mng', 'marketing', 'RandD'))
        salary = st.selectbox('所得水準', options = ('高', '中', '低'))

        # Every form must have a submit button.
        submitted = st.form_submit_button("予測実施！")
        st.write('（ページ右上に結果が表示されます）')

    if submitted:


        if Work_accident == 'あり':
            Work_accident = 1
        else:
            Work_accident = 0

        if promotion_last_5years == 'あり':
            promotion_last_5years = 1
        else:
            promotion_last_5years = 0

        # Encoding salary
        if salary == '高':
            salary = 'high'
        elif salary == '中':
            salary = 'medium'
        else:
            salary = 'low'

        # label encoding
        sales = le.transform([sales])[0]

        # ordinal encoding
        salary = oe.transform(np.array([salary]).reshape(-1, 1))[0][0]

        
        df_pred = pd.DataFrame([
            float(satifsaction_level),
            float(last_evaluation),
            int(number_project),
            int(average_montly_hours),
            int(time_spend_company),
            int(Work_accident),
            int(promotion_last_5years),
            float(sales),
            float(salary)
        ]).T
        df_pred.columns = [
            'satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'sales', 'salary'
        ]

        return df_pred


    # # container sample
    # st.container()

def b_function():
    pass

def prediction_panel():
    st.write("従業員退職予測の実行")


    # columns sample
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write('予測する変数を設定してください')
        df_pred = data_input()

    with col2:
        if df_pred is not None:
            df_pred_processed = ss.transform(df_pred)
            pred = model.predict(df_pred_processed)[0]
            pred_proba = model.predict_proba(df_pred_processed)[0, 1]

            if pred == 0:
                val = '低'
            elif pred == 1:
                val = '高'

            st.metric('従業員退職リスク', val)
            st.metric('従業員退職確率', str(round(pred_proba * 100, 2)) + '%')

            # shap
            df_pred_processed_shap = pd.DataFrame(df_pred_processed, columns = df_pred.columns)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_pred_processed_shap)
            fig2, ax = plt.subplots(figsize = (12, 6))
            shap.summary_plot(shap_values, df_pred_processed_shap, plot_type = 'bar', plot_size=None)
            st.subheader('モデルに対する各変数の寄与度の可視化')
            st.pyplot(fig2)

def main():
    st.title('従業員退職予測モデル')

    mode_list = ["モデルとデータの説明", "従業員退職予測の実行"]
    func_sel = st.sidebar.selectbox("機能を選んでください", mode_list)


    if func_sel == "モデルとデータの説明":
        explorer()

    elif func_sel == "従業員退職予測の実行":
        prediction_panel()


if __name__ == "__main__":
    main()
