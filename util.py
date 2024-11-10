import pandas as pd

input_data =  '2011.01 - 2024.08 VRHCR OPAN Daily Data (결측치가 있는 날짜 제외)'

features = '34개 (VRHCR Feed Property 16개 + 관련 Operation Data 18개)'

feed_property = ['API',
 'Total Sulfur',
 'Total Nitrogen',
 'viscosity 100°C',
 'MCRT',
 'Nickel ',
 'Vanadium',
 'Iron',
 'Sodium',
 'Calcium',
 'SEDEX',
 'Nitrogen/Sulfur ratio',
 'Ni+V',
 'Ni+V_sqrt',
 '4V+Ni']

op_data = ['612FC702.PV', '612TC138B.PV', '612TC138A.PV', '612FC124.PV',
       '612TC168B.PV', '612TC168A.PV', '612FC129.PV', '612TC170A.PV',
       '612FC195A.PV', '612AI111B3.PV/100', '612FI109.PV', '612FI740.PV',
       '612FC123.MV', '612SI101.PV', '612SI102.PV', '612AI104E.PV',
       '612AI107E.PV', 'H2 Partial Pressure']

op_data_new = []
for i in op_data:
    i = i.strip('.PV')
    op_data_new.append(i)

x_columns = ['API',
 'Total Sulfur',
 'Total Nitrogen',
 'viscosity 100°C',
 'MCRT',
 'Nickel ',
 'Vanadium',
 'Iron',
 'Sodium',
 'Calcium',
 'SEDEX',
 'Nitrogen/Sulfur ratio',
 'Ni+V',
 'Ni+V_sqrt',
 '4V+Ni']

feat_imp = [0.014596118591725826,
 0.02278144843876362,
 0.006538881920278072,
 0.008638361468911171,
 0.007710083853453398,
 0.034120168536901474,
 0.013504697009921074,
 0.0056734527461230755,
 0.020842691883444786,
 0.012839130125939846,
 0.005195097532123327,
 0.008018482476472855,
 0.005018230061978102,
 0.0,
 0.00446697510778904]

fi_df = pd.DataFrame({'Feature':x_columns, 'Importance':feat_imp}).sort_values(by='Importance', ascending=False)
fi_df.reset_index(drop=True, inplace=True)
                         

rank_table = pd.DataFrame({
    'Grade': [
        "<span style='color: green;'>👍 Good</span>",   # Green for Good
        "<span style='color: orange;'>⚠️ Bad</span>",  # Orange for Bad
        "<span style='color: red;'>🚫 Worst</span>"    # Red for Worst
    ], 'Description':['SHFT 영향성 낮음','Worst 15 수준 (상위 10%)','Worst 5 수준 (상위 3%)']
})

rank_table_SHFT = pd.DataFrame({
    'Grade': [
        "<span style='color: green;'>👍 Good</span>",   # Green for Good
        "<span style='color: orange;'>⚠️ Bad</span>",  # Orange for Bad
        "<span style='color: red;'>🚫 Worst</span>"    # Red for Worst
    ], 'Description':['SHFT 800 ppm 이하','SHFT 800 ppm 초과','SHFT 1500 ppm 초과']
})