
### 1.討論其他神經網絡表示函數

Week6介紹了奇函數以及週期函數的神經網絡函數表示。在當時試著回答其他課堂中未介紹的函數並具備良好的數學性質的函數，如絕對值函數、階梯函數 (Heaviside Step Function)。


Q:試著討論能不能造出脈衝函數(Dirac Delta Function)

方向一 : 利用極窄的方波函數，$$N(x)=\frac{1}{2ϵ}​[σ(k(x+ϵ))−σ(k(x−ϵ))]$$

方向二：利用極尖的高斯函數，
$$
g(x,σ) = \frac{1}{σ(2\pi)^{1/2}} \exp[-\frac{x^{2}}{2σ^{2}}]
$$
##### AI ans :

方向一 : 

 $N(x)=\frac{1}{2\epsilon}[\sigma(k(x+\epsilon))-\sigma(k(x-\epsilon))]$ 正是**通用逼近定理 (Universal Approximation Theorem)** 證明過程中的核心構造法。

- **學術結論：** 雖然理論上可以逼近，但在數值計算上，Sigmoid 的梯度消失問題會導致極窄的方波難以訓練（Training difficulty）。<br>
    
- **參考文獻：**
    
    - **經典論文 (Cybenko, 1989):** _Approximation by superpositions of a sigmoidal function._ 這篇論文展示了如何用 Sigmoid 的線性組合來構造類似 Delta 的局部函數。
        [連結1 (Link)](https://link.springer.com/article/10.1007/BF02551274)<br>
        
    - **現代應用 (PINNs with Sigmoids):** 在物理信息神經網絡中，研究者使用類似的構造來處理「衝擊波 (Shock waves)」或不連續解。

方向二:

這完全對應到另一種神經網絡架構：**徑向基函數網絡 (Radial Basis Function Networks, RBFN)**。

- **現代應用 - PINNs 中的點源問題 (Point Sources)：** 這是目前最活躍的研究方向。當科學家試圖用 AI 解這類方程式（例如：在一點加熱的熱傳導方程式）時，必須在 Loss Function 中表達 Dirac Delta。直接放入 Delta 會導致電腦數值無限大 (NaN)，因此研究者使用**Mollifier (磨光因子)** 技術，這正是你提到的「讓高斯函數 $\sigma \to 0$」的方法。<br>
    
- **參考文獻：**
        
    - **關鍵論文 (PINNs dealing with Dirac Delta):** 研究如何在神經網絡中處理 Delta 點源。
        - _Moseley, B., et al. (2020). "Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations."_ (雖然主要講區域分解，但處理了局部高頻特徵類似 Delta 的行為)
	    [連結2 (Link)](https://link.springer.com/article/10.1007/s10444-023-10065-9)


### 2.離散資料的 Score Matching

Q:目前 Score Matching 的討論都是基於連續變數（圖片），但我能擴張到離散資料嗎？（文字）。此時梯度 $\nabla_x$ 怎麼辦？

##### AI ans:
- Ratio Matching (比率匹配)：
    
    核心思想是用「差分（Difference）」或「概率比率（Ratio of probabilities）」來取代「微分（Derivative）」。我們不再計算 $\nabla_x \log p(x)$，而是計算 $p(x) / p(\tilde{x})$。
	[連結3 (Link)](https://www.sciencedirect.com/science/article/pii/S0167947306004907)
	    <br>
- Concrete Score Matching / Continuous Relaxation：
    
    將離散數據「連續化」。例如使用 Concrete Distribution (Gumbel-Softmax) 對離散變數進行鬆弛，使其變得可微分，從而可以使用標準的 Score Matching。
    [連結4 (Link)](https://arxiv.org/abs/2211.00802)
    <br>
- Discrete Diffusion (D3PM)：
    
    最新的生成模型（如 D3PM）直接在離散狀態空間定義擴散過程。這裡的「Score」被重新定義為轉移機率的某種形式，而不是幾何上的梯度。
    [連結5 (Link)](https://arxiv.org/abs/2107.03006)
    <br>
### 3. Multimodal 分布與 Failure Modes

Q:如果我的 data distribution 是多個 multimodal（多模態），是不是會生成 Failure Modes？換句話說，產生 Failure Modes 的可能有哪些？

##### AI ans:

主要的 Failure Modes 有以下三種：

1. **Mode Collapse / Mode Dropping (模態丟失)：**
    
    - **現象：** 模型只學會了數據中的一部分模態，而忽略了其他模態。例如數據集有「貓」和「狗」，但模型最後只生成「狗」。
        
    - **原因：** 這通常是上述「混合緩慢」或「分數估計不準」的後果。因為採樣過程無法到達某些區域，或者該區域的梯度資訊遺失，導致模型最終無法生成該類別的數據。<br>
2. **Spurious Modes / Ghost Modes (虛假模態)：**
    
    - **現象：** 模型在兩個真實模態之間生成了不該存在的數據（連線區域）。例如在「兩個分開的圓」之間生成了一條線。
        
    - **原因：** 在低密度區域（Low density regions），由於缺乏訓練數據，Score function (梯度場) 的估計不準確，導致梯度指向錯誤，將兩個原本分開的流形（Manifold）錯誤地連接在一起。<br>
3. **Slow Mixing (混合緩慢)：**
    
    - **現象：** 雖然模型學會了所有模態，但採樣過程（Langevin Dynamics）卡在某一個模態出不來，無法遍歷所有可能的結果。
        
    - **原因：** 不同模態之間被「低機率密度區域」隔開。Langevin Dynamics 就像一個爬山的人，很難跨越深谷去爬另一座山，跨越這些區域所需的步數（Mixing time）是指數級增長的。<br>

**參考文獻 (Reference)：**

這是一個已被廣泛研究的問題，解決方案（如 Noise Perturbation）也是該領域的基石。

- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. _Advances in Neural Information Processing Systems (NeurIPS)_.
  [連結6 (Link)](https://arxiv.org/abs/1907.05600)

### 4. SDE 理論在深度學習應用中的廣義化（Generalization)

Q:目前 SDE 是建立在 Wiener Process 上，那我的數據結構若不適合這種噪音結構（例如非歐幾里得數據、或非高斯噪聲），我能不能擴展下去？

##### AI ans:

目前的學術研究已經將 SDE 擴展到了兩種主要的情況，以解決標準 Wiener Process 不適用的問題：

- 1. **數據結構不適合（非歐幾里得空間）→ 黎曼流形 SDE (Riemannian SDEs)**
    
    - 如果你的數據不是分佈在平面上，而是分佈在球面上（如地球科學數據）或特定的幾何結構上（如機器人關節角度、蛋白質結構），標準的 Wiener Process 會失效。
        <br>
    - **解決方案：** 研究人員引入了 **Riemannian Score-Based Generative Models**。這種方法將 SDE 定義在黎曼流形上，使用「布朗運動在流形上的推廣（Geodesic Random Walk）」，讓擴散過程沿著幾何表面進行，而非直線進行。
	    <br>
	- **參考文獻 :**  De Bortoli, V., et al. (2022). Riemannian Score-Based Generative Modelling. _Advances in Neural Information Processing Systems (NeurIPS)_.
	  [連結7 (Link)](https://arxiv.org/abs/2202.02763)
	  <br>
- 2. **噪音結構不適合（非高斯噪聲）→ 萊維過程 (Lévy Processes)**
    
    - Wiener Process 假設噪音是連續且服從高斯分佈的。但如果你的數據包含劇烈的突變（Jumps）或長尾分佈（Heavy tails）（例如金融市場數據、醫學影像中的異常點），高斯噪聲就不適用。
        <br>
    - **解決方案：** 可以使用 **Lévy Processes** 來取代 Wiener Process。這類 SDE 允許引入「跳躍（Jumps）」項，使得模型可以處理非連續變化的噪聲結構。
	    <br>
	- **參考文獻 :**  Yoon, E. B., et al. (2023). Score-based Generative Models with Lévy Processes. _Advances in Neural Information Processing Systems (NeurIPS)_.
	  [連結8 (Link)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html)

### 5. 關於 PF-ODE 中 $g(x,t)$ 的依賴項

Q:在 PF-ODE 中，若擴散係數 $g$ 依賴於 $x$（即 $g(x,t)$），多出來的 $\frac{\partial}{\partial x} g^2(x,t)$ 項在實作上該如何學習？

##### AI ans:

這是一個觀念上的誤區，這項其實**不需要「學習」**，而是透過**「解析計算（Analytical Calculation）」**或**「自動微分（Auto-differentiation）」**來處理。

**參考文獻 (Reference)**

- **論文：** Score-Based Generative Modeling through Stochastic Differential Equations (ICLR 2021)
   [連結9(Link)](https://arxiv.org/abs/2011.13456)
   <br>
- **說明：** Yang Song 在這篇論文中推導 PF-ODE 時，明確指出了 forward SDE 的 $f$ 和 $g$ 是由設計者選擇的（user-specified），因此其導數項是已知的，唯有 $\nabla \log p$ 需要估計。

### 6. Sampling Efficiency & Adaptive Step Size

Q：在擴散模型反向過程中，越接近 $t=0$ 梯度越陡峭。固定 $\Delta t$ 很難兼顧效率與精準度。那我們有辦法做個非線性的 $\Delta t$ 來達到省時跟精準的優勢嗎？

##### AI ans:

這是一個**已解決且高度成熟**的技術領域，通常被稱為「**自適應步長 ODE 求解器 (Adaptive Step-Size ODE Solvers)**」。

- 1. 通用數值解法 (Adaptive Solvers)：
    - 我們不需要手動設計「非線性 $\Delta t$」。在解 PF-ODE 時，可以直接使用成熟的數值方法（如 Runge-Kutta 45 / Dormand-Prince method）。這些算法會根據當前的「誤差估計（Error Tolerance）」自動調整每一步的 $\Delta t$。
    <br>
    - 當梯度變化劇烈（如 $t \to 0$），求解器會自動縮小 $\Delta t$ 以保證精確 ; 當梯度平緩，求解器會放大 $\Delta t$ 以節省時間。
    <br>
- 2. 專用的加速算法 (DPM-Solver)：

    - 除了通用的 ODE Solver，學術界更發展出了針對擴散模型的專用求解器，如 DPM-Solver。它利用了擴散過程的半線性結構（Semi-linear structure），能在極少的步數（如 10~20 步）內完成高品質採樣，這本質上就是在做最高效的非線性步長規劃。
      <br>
    -  **參考文獻 :**  Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. _Advances in Neural Information Processing Systems (NeurIPS)_.
	  [連結10 (Link)](https://arxiv.org/abs/2206.00927)
    