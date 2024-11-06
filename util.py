import pandas as pd

input_data =  '2011.01 - 2024.08 VRHCR OPAN Daily Data (결측치가 있는 날짜 제외)'

features = 'VRHCR Feed Property (10개) + 관련 Operation Data (19개)'

feed_property = ['API',
       'Total Sulfur', 'Total Nitrogen', 'viscosity @100°C', 'MCRT', 'Nickel ',
       'Vanadium', 'Iron', 'Sodium', 'Calcium']

op_data = ['612FI747.PV', '612FC702.PV', '612FQI116.PV', '612TC138B.PV',
       '612TC138A.PV', '612PI131A.PV', '612FC123.PV', '612FC124.PV',
       '612FC125.PV', '612TC168B.PV', '612TC168A.PV', '612FC127A.PV',
       '612FC127B.PV', '612FC129.pv', '612TC170A.PV', '612PC148A.pv',
       '612TI707.PV', '612FC195A.PV', '612AI111B3.PV']
op_data_new = []
for i in op_data:
    i = i.strip('.pv')
    i = i.strip('.PV')
    op_data_new.append(i)
op_data_new

x_columns = ['Gravity, API', 'Total Sulfur', 'Total Nitrogen', 'vis 100°C', 'MCRT',
       'Nickel ', 'Vanadium', 'Iron', 'Sodium', 'Calcium']

feat_imp = [0.01498538, 0.02591298, 0.00766891, 0.00961473, 0.01018605,
       0.03045351, 0.00740778, 0.00478776, 0.02919257, 0.0090797]

fi_df = pd.DataFrame({'Feature':x_columns, 'Importance':feat_imp}).sort_values(by='Importance', ascending=False)

# rank_table = pd.DataFrame({'등급':['👍 Good','⚠️ Bad','🚫 Worst'], '설명':['원유 비율 30% 에서 SHFT 0.10 % 이하','원유 비율 0 - 30%에서 SHFT 0.10 % 초과','원유 비율 5%에서 SHFT 0.15 % 이상']})                           

rank_table = pd.DataFrame({
    'Grade': [
        "<span style='color: green;'>👍 Good</span>",   # Green for Good
        "<span style='color: orange;'>⚠️ Bad</span>",  # Orange for Bad
        "<span style='color: red;'>🚫 Worst</span>"    # Red for Worst
    ], '     Description    ':['원유 비율 30% 에서 SHFT 0.10 % 이하','원유 비율 0 - 30%에서 SHFT 0.10 % 초과','원유 비율 5%에서 SHFT 0.15 % 이상']
})

