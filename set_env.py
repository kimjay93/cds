import subprocess

print('='*60)
print('기본 환경 구성 중입니다.')
print('='*60)

packages = {
    'scikit-learn':'1.4',
    'xgboost':'2.1.1',
    'lightgbm':'4.5.0',
    'catboost':'1.2.6',
    'optuna':'4.0.0'
}

if len(packages) > 0:
    for key in packages:
        subprocess.run(['pip', 'install', key+'=='+packages[key]])


print('='*60)
print('기본 환경 구성을 완료했습니다.')
print('='*60)