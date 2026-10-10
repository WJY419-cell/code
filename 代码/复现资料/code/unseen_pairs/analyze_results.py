from pathlib import Path
import csv,json,math,os
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import wilcoxon
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[2]/'tmp/matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=Path(os.environ.get('MPC_UNSEEN_OUT',str(ROOT/'supplement/unseen_pairs'))).expanduser()
def read(name):return list(csv.DictReader((P/name).open(encoding='utf-8-sig')))
def write(name,rows):
 with (P/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
 done=json.loads((P/'completion.json').read_text());assert done['scenarios']==216 and done['rows']==1296
 rows=read('per_seed.csv');audit=read('split_audit.csv');groups=defaultdict(list)
 for r in rows:groups[(r['dataset'],r['mode'],float(r['fraction']),r['method'])].append(r)
 summary=[]
 for (ds,mode,frac,method),rr in sorted(groups.items()):
  assert len(rr)==12 and len({r['seed'] for r in rr})==12
  record=dict(dataset=ds,mode=mode,fraction=frac,method=method,n_runs=12)
  for metric in ['test_nll','accuracy']:
   x=np.array([float(r[metric]) for r in rr]);record[metric+'_mean']=float(x.mean());record[metric+'_std']=float(x.std(ddof=1));record[metric+'_ci95']=float(1.96*x.std(ddof=1)/np.sqrt(12))
  summary.append(record)
 write('summary.csv',summary)
 tests=[];rng=np.random.default_rng(20261005)
 for (ds,mode,frac,method),rr in sorted(groups.items()):
  if method!='MPC-Rank':continue
  x=np.array([float(r['test_nll']) for r in sorted(rr,key=lambda r:int(r['seed']))])
  for base in ['Pooled-BT','Layerwise-BT','Uncoupled-Joint','RankCentrality','Borda']:
   yy=groups[(ds,mode,frac,base)];y=np.array([float(r['test_nll']) for r in sorted(yy,key=lambda r:int(r['seed']))]);d=y-x
   dstat=d.copy();dstat[np.abs(dstat)<1e-8]=0
   ps=1. if not np.any(dstat) else float(wilcoxon(dstat,alternative='greater').pvalue)
   boot=d[rng.integers(0,12,(5000,12))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
   tests.append(dict(dataset=ds,mode=mode,fraction=frac,baseline=base,mean_nll_gain=float(d.mean()),relative_nll_reduction=100*(y.mean()-x.mean())/y.mean(),wins=int(np.sum(dstat>0)),ties=int(np.sum(dstat==0)),losses=int(np.sum(dstat<0)),bootstrap_low=float(lo),bootstrap_high=float(hi),p_one_sided=ps))
 assert len(tests)==90
 order=sorted(range(len(tests)),key=lambda t:tests[t]['p_one_sided']);running=0.
 for rank,index in enumerate(order):
  running=max(running,(len(tests)-rank)*tests[index]['p_one_sided']);tests[index]['p_holm_all90']=min(1.,running)
 write('paired_effects.csv',tests)
 dslist=['Breakfast-6Context','SUSHI-RegionPaths','MovieLens-OccupationPaths'];short=['Breakfast','SUSHI','MovieLens'];methods=['Pooled-BT','Layerwise-BT','Uncoupled-Joint','MPC-Rank','RankCentrality','Borda']
 fig,axs=plt.subplots(2,3,figsize=(12,6.4),sharex=True)
 colors=['#49925B','#E78655','#537999','#8B59C2','#464646','#B59B3A']
 for row,mode in enumerate(['within_layer','global']):
  for col,ds in enumerate(dslist):
   ax=axs[row,col]
   for method,color in zip(methods,colors):
    rs=sorted([r for r in summary if r['dataset']==ds and r['mode']==mode and r['method']==method],key=lambda r:r['fraction'])
    x=[100*r['fraction'] for r in rs];y=np.array([r['test_nll_mean'] for r in rs]);ci=np.array([r['test_nll_ci95'] for r in rs])
    ax.plot(x,y,'o-',color=color,label=method,ms=3,lw=1.5);ax.fill_between(x,y-ci,y+ci,color=color,alpha=.07)
   ax.set_title(short[col]+' / '+('within-layer unseen' if row==0 else 'globally unseen'),fontsize=10);ax.set_xticks([10,20,30]);ax.grid(alpha=.2);ax.set_ylabel('Test NLL (lower is better)')
   if row==1:ax.set_xlabel('Masked pairs (%)')
 fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=6,loc='upper center',frameon=False,fontsize=9)
 fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(P/'unseen_pair_results.png',dpi=200);plt.close(fig)
 text=['# 未见对象对实验：结果与分析','日期：2026-10-05。论文原文、论文 PDF 及原模型代码保持未改动。',
 '## 主要判断',
 '**新增实验确实补上了 unseen item pairs 的实证任务，但没有得到“图耦合全面优于基线”的结果。** 全局遮蔽下，测试对象对在所有层训练中均未直接出现；层内遮蔽下，同层没有该对象对的训练比较。原论文未做这类隔离，现在有了单独、可核验的新证据。',
 '**Breakfast：相对 pooling 的收益稳定，不能归因于图拓扑。** 主设置两种范围相对 pooled 的 NLL 分别降低 4.01% 和 4.07%，均为 12/12 种子改善；但比独立模型和无图联合模型差约 0.47%/0.51%。两种范围都在 10/12 次验证中选择零耦合。均匀完全图本身又等价于额外 ridge，因此这里主要支持上下文分数的价值，不支持非均匀图共享。当前等层样本量且 λ_b=λ_u/K 的特殊配置还使零耦合联合目标等价于分层 ridge 拟合，解释了无图联合与独立模型数值接近。',
 '**SUSHI：层内未见是最明确的结构共享证据，但没有超过所有基线。** 主设置相对独立估计和无图联合模型分别降低 NLL 0.47% 和 0.29%；12 个种子均选择非零耦合。10%—30% 遮蔽下，相对无图的收益维持 0.25%—0.32%。但对 pooled 的差异接近零（主设置反而高 0.03%），Rank Centrality 更好。全局未见时图耦合不优于无图联合模型，说明该收益有明确范围。',
 '**MovieLens：两种范围都超过 pooled 和 layerwise，但图正则没有显示额外收益。** 主设置层内 NLL 分别改善 0.34%/0.95%，全局改善 0.49%/1.04%；这些方向在 10%—30% 遮蔽下保持一致。然而无图联合模型的平均 NLL 在全部六种设置都更低，全局主设置 10/12 次选择零耦合。不能将完整模型相对 pooled/layerwise 的优势直接说成图耦合的优势。Rank Centrality 和 Borda 的 NLL 也低于 MPC，故不构成最佳方法证明。',
 '**统计与实际意义要分开。** 主设置 Breakfast/MovieLens 相对 pooled、SUSHI 层内相对独立/无图的有利比较，在本次 90 项比较的 Holm 校正后仍有 p<0.05；SUSHI 全局相对 pooled 的小收益不通过该校正。这些是有限数据重拆分的描述性证据，效果量大多较小，不能据此推断跨人群普遍优势。',
 '**对论文的意义：可以补强“预测未见对象对”的实验依据，但不能用它替换原低支持目标层结果。** 遮蔽对象对与减少目标层用户记录是不同稀疏机制。新实验未证明后者现在也超过 pooling；若未来纳入论文，应保留原实验的相反结果并区分两类缺失。当前不修改论文，也不选择性只保留胜出的数据集或设置。',
 '## 设计与解释范围',
 '沿用三个数据集、模型公式、ridge 系数、原用户/记录级外层拆分及每层训练记录上限。每种情形使用种子 0—11；20% 遮蔽为预先指定主设置，10% 和 30% 为敏感性设置。共 3 数据集 × 2 遮蔽范围 × 3 比例 × 12 种子 = 216 个情形，每个情形比较 6 个方法。',
 '- 层内未见：每层独立抽取对象对，将其在该层的全部训练/验证比较移除；其他层可保留。测试仅使用原外层测试用户对这些对象对的比较。',
 '- 全局未见：各层使用同一对象对掩码，选中对象对在所有层的训练/验证中均被移除。测试对象对在最终训练集合中的全局交集为零。',
 '- 掩码由固定对象目录与随机种子产生，不查看胜负方向或实验表现；同一种子不同遮蔽比例使用嵌套子集，不筛掉效果差的种子。',
 '- 调参另划内层用户/记录，并在剩余对象对上生成同范围的独立遮蔽集合。仅用这些内层未见对象对的验证 NLL 选择 λ_s；最终再用全部外层训练记录拟合（继续移除外层测试对象对）。',
 '- MovieLens 图仅用当前训练记录的物品共同出现构建，不读取评分方向或相等与否；被遮蔽对象对也不进入图覆盖向量。内层图仅由内层训练构建。物品目录仍沿用原固定目录，因此不宣称全流程归纳式新物品预测。',
 '- 所有方法的训练和测试比较完全一致。Pooled-BT/Layerwise-BT 沿用 λ_u=0.03；联合模型 λ_u=0.03、λ_b=0.005；MPC 从 {0,0.01,0.05,0.2,0.8} 选择 λ_s。无图联合模型为 λ_s=0。Borda/RankCentrality 沿用训练集温度校准。没有对所有基线重新进行同等规模的 ridge 搜索，所以结论限定于这些设置。',
 '## 主设置：遮蔽 20% 对象对',
 'NLL 越低越好；相对改善=100×(基线平均 NLL−MPC 平均 NLL)/基线平均 NLL，负值表示 MPC 更差。',
 '|范围|数据|Pooled|Layerwise|无图联合|MPC|Rank Centrality|Borda|','|---|---|---:|---:|---:|---:|---:|---:|']
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   vals=[next(r['test_nll_mean'] for r in summary if (r['dataset'],r['mode'],r['fraction'],r['method'])==(ds,mode,.2,m)) for m in methods]
   text.append('|'+('层内' if mode=='within_layer' else '全局')+'|'+sh+'|'+'|'.join(f'{v:.5f}' for v in vals)+'|')
 text+=['','|范围|数据|相对 pooled|相对 layerwise|相对无图|优于 pooled 的种子数|','|---|---|---:|---:|---:|---:|']
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   rs=[next(r for r in tests if (r['dataset'],r['mode'],r['fraction'],r['baseline'])==(ds,mode,.2,b)) for b in methods[:3]]
   text.append('|'+('层内' if mode=='within_layer' else '全局')+'|'+sh+'|'+'|'.join(f"{r['relative_nll_reduction']:+.2f}%" for r in rs)+f"|{rs[0]['wins']}/12|")
 text+=['','## 次要指标：主设置准确率','NLL 改善不自动意味着准确率改善；下面保留所有方法的准确率。','|范围|数据|Pooled|Layerwise|无图联合|MPC|Rank Centrality|Borda|','|---|---|---:|---:|---:|---:|---:|---:|']
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   vals=[next(r['accuracy_mean'] for r in summary if (r['dataset'],r['mode'],r['fraction'],r['method'])==(ds,mode,.2,m)) for m in methods]
   text.append('|'+('层内' if mode=='within_layer' else '全局')+'|'+sh+'|'+'|'.join(f'{v:.2%}' for v in vals)+'|')
 text+=['','## 10%/30% 敏感性：相对 NLL 改善','|范围|数据|遮蔽|相对 pooled|相对 layerwise|相对无图|','|---|---|---:|---:|---:|---:|']
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   for fraction in [.1,.2,.3]:
    rs=[next(r for r in tests if (r['dataset'],r['mode'],r['fraction'],r['baseline'])==(ds,mode,fraction,b)) for b in methods[:3]]
    text.append('|'+('层内' if mode=='within_layer' else '全局')+'|'+sh+f'|{fraction:.0%}|'+'|'.join(f"{r['relative_nll_reduction']:+.2f}%" for r in rs)+'|')
 text+=['','## 数据隔离及数值检查']
 assert all(int(r['leakage_violations'])==0 for r in audit)
 assert all(float(r['donor_coverage_fraction'])==0 for r in audit if r['mode']=='global')
 fits=json.loads((P/'optimizer_checks.json').read_text());gmax=max(r['gradient_inf'] for r in fits)
 assert gmax<=2e-6
 text.append(f"216 个情形的掩码隔离断言均通过，用户/记录无交叉；全局未见测试对象对的训练覆盖率均为 0。{len(fits):,} 次求解均通过一阶残差验收，最大残差 {gmax:.3g}。原论文和求解器哈希核对不变。")
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   rr=[r for r in audit if r['mode']==mode and r['dataset']==ds and float(r['fraction'])==.2]
   text.append(f"- {sh} / {mode} / 20%：测试比较数 {min(float(r['test_comparisons']) for r in rr):.0f}—{max(float(r['test_comparisons']) for r in rr):.0f}；其他层训练中同对象对覆盖率均值 {np.mean([float(r['donor_coverage_fraction']) for r in rr]):.1%}；存在不连通层的拆分数 {sum(int(r['disconnected_layers'])>0 for r in rr)}/12。")
 text+=['','## 图耦合选择与推断边界']
 for mode in ['within_layer','global']:
  for ds,sh in zip(dslist,short):
   vals=Counter(float(r['lambda_s']) for r in rows if r['mode']==mode and r['dataset']==ds and float(r['fraction'])==.2 and r['method']=='MPC-Rank')
   text.append(f'- {sh} / {mode}：主设置 λ_s 次数 {dict(sorted(vals.items()))}。')
 text+=['','结果区间和配对检验描述同一有限数据上的重复拆分，不是独立人群重复。`paired_effects.csv` 同时提供 5000 次拆分级配对 bootstrap 区间、单侧 Wilcoxon p 值和全部 90 次 NLL 比较的 Holm 校正。比较对之间、同一排序诱导的比较之间并不独立；不能将比较总数当作独立样本量。不得将最小测试 NLL 的方法事后当作已经部署的选择策略。差值绝对值小于 1e-8 在胜负计数和秩检验中视为数值平局；平均效果和 bootstrap 保留原始差值。',
 '曲线中实线为 12 次拆分的算术均值，阴影为均值 ±1.96×标准差/√12。每次拆分内部 NLL 按比较次数加权，跨拆分均值不再按比较总量重新加权。层内和全局模式的测试对象对分布不完全相同，不能直接用两个模式的绝对 NLL 大小证明哪一种任务更容易；应比较各模式内部相同测试集上的方法差异。MovieLens 各拆分存在不连通的层内对象图，缺失方向的估计依赖正则和共享假设；这些拆分全部保留并披露，没有筛除。',
 '仅保留部分比较后，剩余比较仍可能通过传递关系间接约束被遮蔽对象对；这属于模型可用信息而不是直接比较泄漏。本实验验证“训练未直接观察对象对”，并非“没有任何间接信息”。全局未见也不等于物品本身从未出现。',
 '三个数据集的原实验与本实验不构成只改变遮蔽的纯因果对照：本实验还使用模拟未见对象对的内层调参，MovieLens 使用更严格的训练图。跨数据集和跨协议的收益不能合并宣称普遍优势。',
 '', '## 文件', '`protocol.json` 为运行前写入的设计；`per_seed.csv`、`summary.csv`、`paired_effects.csv`、`tuning.csv` 保存完整结果；`split_audit.csv` 与 `pair_masks.npz` 保存隔离诊断与全部掩码；`optimizer_checks.json` 保存求解残差。`unseen_pair_results.png` 为全比例曲线。实验代码位于项目 experiments/unseen_pairs_20261005。']
 (P/'实验结果与分析.md').write_text('\n\n'.join(text).replace('|\n\n|','|\n|'),encoding='utf-8')
 print(json.dumps([r for r in tests if r['fraction']==.2],indent=2))
 print('FITS',len(fits),'MAX',gmax)
if __name__=='__main__':main()
