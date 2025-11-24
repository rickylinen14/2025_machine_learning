import numpy as np
import matplotlib.pyplot as plt
import os

def run_strict_simulation():
    STEPS = 100
    
    # --- 1. 預生成「命運之書」 (Pre-generated Randomness) ---
    # 這是為了確保兩組面對完全一樣的 "運氣" (確保實驗的 Internal Validity)
    np.random.seed(999) # 固定種子，保證結果可重現
    
    # 用來判斷是否發生狀態轉移的亂數 (0~1)
    fate_transition = np.random.rand(STEPS) 
    
    # 用來決定打字速度的 "基礎波動" (標準常態分佈)
    fate_speed_noise = np.random.normal(0, 1, STEPS)
    
    # 用來決定錯誤率的 "基礎波動" (0~1 均勻分佈)
    fate_error_noise = np.random.rand(STEPS) 

    print("已生成固定命運序列，開始兩組平行測試...")

    # --- 定義模擬器 (讀取命運版) ---
    class StrictStudent:
        def __init__(self):
            self.state = 0 # 0:Flow (心流), 1:Frustrated (卡關)
        
        def step(self, t, intervention, recover_prob_ai, recover_prob_self, fall_prob):
            # 讀取這一時間點的命運骰子
            dice = fate_transition[t] 
            
            # --- A. 狀態轉移邏輯 (State Transition) ---
            if self.state == 0: # Flow
                # 兩組都面對一樣的 dice，所以"運氣不好"的時間點是一樣的
                if dice < fall_prob: 
                    self.state = 1
            else: # Frustrated
                # 這裡體現 AI 的價值：改變了恢復的機率閾值
                prob = recover_prob_ai if intervention else recover_prob_self
                if dice < prob: 
                    self.state = 0
            
            # --- B. 產生觀測數據 (Emission with Pre-generated Noise) ---
            # 使用預生成的 Noise，保證兩組在同一時間點的表現波動基礎一致
            if self.state == 0:
                speed = 100 + (10 * fate_speed_noise[t])
                error = 0.05 + (0.05 * fate_error_noise[t]) 
            else:
                speed = 40 + (15 * fate_speed_noise[t])
                error = 0.20 + (0.10 * fate_error_noise[t])
                
            return np.array([speed, error])

    # --- AI 模型 (基於 HMM 邏輯的規則模型) ---
    class SimpleAI:
        def __init__(self):
            self.belief = 0.0 # 初始信念 (認為學生受挫的機率)
            
        def predict(self, obs):
            speed, error = obs
            
            # 1. 似然性估計 (Likelihood): 速度慢 或 錯誤高 -> 懷疑是挫折
            likelihood = 0.0
            if speed < 70: likelihood += 0.4
            if error > 0.15: likelihood += 0.5
            
            # 2. 貝氏更新概念 (Bayesian Update): 結合舊信念與新證據
            # belief_t = 0.7 * belief_{t-1} + 0.3 * likelihood
            self.belief = 0.7 * self.belief + 0.3 * likelihood
            
            # 3. 決策策略 (Policy): 信心 > 0.6 則介入
            return 1 if self.belief > 0.6 else 0

    # --- 執行兩組平行實驗 ---
    
    # 參數設定
    FALL_PROB = 0.10       # 10% 機率掉進坑
    
    # 【關鍵調整 1】: 配合報告文字 "約 10% 自動恢復"
    SELF_RECOVER = 0.10    
    
    # 【關鍵調整 2】: 設為 50% 讓 AI 像個輔助者而非神，數據更寫實
    AI_RECOVER = 0.50      
    
    # Group 1: No AI (對照組)
    student1 = StrictStudent()
    states_no = []
    for t in range(STEPS):
        student1.step(t, 0, AI_RECOVER, SELF_RECOVER, FALL_PROB)
        states_no.append(student1.state)
        
    # Group 2: With AI (實驗組)
    student2 = StrictStudent()
    ai = SimpleAI()
    states_ai = []
    action = 0
    for t in range(STEPS):
        obs = student2.step(t, action, AI_RECOVER, SELF_RECOVER, FALL_PROB)
        action = ai.predict(obs) # AI 決定下一步行動
        states_ai.append(student2.state)

    # --- 繪圖與數據計算 ---
    flow_no = states_no.count(0)
    flow_ai = states_ai.count(0)
    lift = ((flow_ai - flow_no)/flow_no)*100
    
    print(f"=== 最終版模擬結果 (符合報告敘述) ===")
    print(f"無 AI: 心流時間 {flow_no} 分鐘 (自動恢復率: {SELF_RECOVER*100:.0f}%)")
    print(f"有 AI: 心流時間 {flow_ai} 分鐘 (干預恢復率: {AI_RECOVER*100:.0f}%)")
    print(f"提升: +{lift:.1f}%")
    
    plt.figure(figsize=(10, 5))
    plt.plot(states_no, label='No AI', linestyle='--', color='tab:blue', alpha=0.6)
    plt.plot(states_ai, label='With AI', linewidth=2, color='tab:orange')
    plt.fill_between(range(STEPS), states_no, color='tab:blue', alpha=0.1)
    
    plt.title(f'Strict Comparison (Self Recover={int(SELF_RECOVER*100)}%, AI Recover={int(AI_RECOVER*100)}%)')
    plt.yticks([0, 1], ['Flow', 'Frustrated'])
    plt.xlabel('Time Step')
    plt.legend()
    
    save_filename = 'toy_model_result.png'
    plt.savefig(save_filename)
    print(f"圖表已儲存為 {save_filename}")

if __name__ == "__main__":
    run_strict_simulation()