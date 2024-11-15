import pandas as pd
df = pd.read_csv('./df_58thtrial.csv', index_col = 'Datetime')

op_data = ['612FC702.PV', '612TC138A.PV', '612FC124.PV', '612TC168A.PV',
       '612FC129.PV', '612TC170A.PV', '612FC195A.PV', '612FI109.PV',
       '612FI740.PV', '612FC123.MV', '612SI101.PV', '612SI102.PV',
       '612AI104E.PV', '612AI107E.PV', 'H2 Partial Pressure']
op_data_new = []
for i in op_data:
    i = i.strip('.PV')
    op_data_new.append(i)

feed_property = ['API', 'Sulfur %',
       'Nitrogen ppm', 'MCRT wt. %', 'Ni ppm', 'V ppm', 'Na ppm', 'Fe ppm',
       'Ca ppm', 'V(100) cSt.', 'ASP(C7) wt. %']

input_data = '2012.01 - 2024.08 VRHCR OPAN Daily Data (결측치가 있는 날짜 제외)'

features = '26개 (VRHCR Feed Property 11개 + 관련 Operation Data 15개)'

feat_imp_var = ['API',
 'Sulfur %',
 'Nitrogen ppm',
 'MCRT wt. %',
 'Na ppm',
 'Fe ppm',
 'Ca ppm',
 'ASP(C7) wt. %',
 'vis100_sqrt',
 'Ni_V_PCA']

feat_imp = [0.00648,
 -0.00323,
 -0.00042,
 0.00066,
 -0.00131,
 0.00169,
 -0.00068,
 0.00129,
 0.00238,
 -5e-05]

fi_df = pd.DataFrame({'Feature':feat_imp_var, 'Importance':feat_imp}).sort_values(by='Importance', ascending=False)
fi_df.reset_index(drop=True, inplace=True)
                         

rank_table = pd.DataFrame({
    'Grade': [
        "<span style='color: green;'>🟢 Good</span>",   # Green for Good
        "<span style='color: yellow;'>🟡 Moderate</span>",   # Yellow for Moderate
        "<span style='color: orange;'>🟠 Bad</span>",  # Orange for Bad
        "<span style='color: red;'>🔴 Worst</span>"    # Red for Worst
    ], 'SHFT 영향성':['하위 50%', '상위 50% 이상', '상위 10% 이상 (Worst 15)','상위 3% 이상 (Worst 5)']
})

rank_table_SHFT = pd.DataFrame({
    'Grade': [
        "<span style='color: green;'>🟢 Normal</span>",   # Green for Good
        "<span style='color: orange;'>🟠 High</span>",  # Orange for Bad
        "<span style='color: red;'>🔴 Max</span>"    # Red for Worst
    ], 'SHFT 기준':['800 ppm 이하', '800-1500 ppm' ,'1500 ppm 초과']
})