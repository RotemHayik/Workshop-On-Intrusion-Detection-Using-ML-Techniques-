from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import argparse
parser = argparse.ArgumentParser(description='Recreate the six submission figures in a new folder.')
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
r = args.root
out = args.output
out.mkdir(parents=True, exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors=['#255F85','#438A70','#B07736','#79619C','#B34949']
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color='#EAF1F5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',facecolor=color,edgecolor='#8A9CAB',lw=1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526572','lw':1.4})
fig,ax=plt.subplots(figsize=(10.7,5.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.70,.21,.19,'Ingestion adapter\nCanonical DNS transactions\nFrozen recording splits')
box(ax,.30,.70,.27,.19,'Train only preprocessing\nRobust scaling and imputation\n128 transactions per window')
box(ax,.65,.70,.32,.19,'Three aligned representations\n42 tabular features\n128 × 15 sequence + categories\nWithin-window entity graphs')
arrow(ax,(.23,.79),(.30,.79));arrow(ax,(.57,.79),(.65,.79))
box(ax,.04,.36,.19,.19,'Random Forest\nSupervised\nClass and group weights')
box(ax,.29,.36,.19,.19,'Isolation Forest\nBenign Train only\nBalanced recordings')
box(ax,.54,.36,.19,.19,'Bidirectional LSTM\nCompleted sequences\nBalanced sampling')
box(ax,.79,.36,.19,.19,'GraphSAGE\nTopology + nodes\nBalanced sampling')
for x in [.135,.385,.635,.885]:arrow(ax,(.81,.69),(x,.56))
box(ax,.10,.035,.36,.17,'Validation only\nSensitivity and early stopping\nFreeze models and thresholds')
box(ax,.56,.035,.37,.17,'Held-out Test and External Test\nMetrics · source errors · cascade\nLocal LLM arbitration')
for x in [.135,.385,.635,.885]:arrow(ax,(x,.35),(.28,.215))
arrow(ax,(.46,.12),(.56,.12));save(fig,'01_modular_pipeline')

fig,ax=plt.subplots(figsize=(10.7,5.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.66,.23,.23,'Stage 1\nRF + refined IF\nExtreme RF score and\nmatching decisions?')
box(ax,.35,.66,.26,.23,'Stage 2 for remaining windows\nBiLSTM + GraphSAGE\nMajority of supervised models')
box(ax,.72,.66,.26,.23,'Local Qwen arbitrator\nAt most 32 windows per split\nAt most 2 per recording\nHash order independent of label')
box(ax,.33,.13,.33,.25,'Final prediction and evidence\nUse valid LLM label when selected\nOtherwise keep cascade decision\nLog all routing and fallback counts')
arrow(ax,(.25,.78),(.35,.78));ax.text(.3,.83,'No',ha='center')
arrow(ax,(.61,.78),(.72,.78))
arrow(ax,(.13,.65),(.33,.26));ax.text(.10,.43,'Yes: RF',ha='center')
arrow(ax,(.85,.65),(.66,.26));arrow(ax,(.47,.65),(.47,.39))
ax.text(.50,.50,'Budget or response fallback',ha='left',fontsize=9)
ax.text(.5,.03,'Validation froze the override rule before Test; held-out harm is reported without retuning.',ha='center',fontsize=10)
save(fig,'02_behavioral_cascade')

m=pd.read_csv(r/'06_results/performance_matrix.csv');names=['rf_base','bilstm_base','sage_base','if_base','if_no_volume_timing'];labels=['RF','BiLSTM','GraphSAGE','IF baseline','IF refined']
z=m[(m.split=='test')&(m.level=='window')].set_index('run_id').loc[names]
fig,axes=plt.subplots(1,2,figsize=(10.3,3.4),gridspec_kw={'width_ratios':[1.3,1]})
axes[0].barh(labels,z.recall*100,color=colors);axes[0].set(xlim=(85,101),xlabel='Malicious recall (%)');axes[0].invert_yaxis()
for i,v in enumerate(z.recall*100):axes[0].text(v+.25,i,f'{v:.2f}',va='center',fontsize=9)
x=np.arange(5);axes[1].bar(x-.18,z.fp,width=.36,label='False positives',color='#255F85');axes[1].bar(x+.18,z.fn,width=.36,label='False negatives',color='#B34949');axes[1].set(xticks=x,xticklabels=labels,ylabel='Window count');axes[1].tick_params(axis='x',labelrotation=25);axes[1].legend(frameon=False,fontsize=9)
fig.suptitle('Held-out Test · 1,179 benign and 1,160 malicious windows',fontsize=12);fig.tight_layout();save(fig,'03_internal_test_performance')

fig,axes=plt.subplots(1,5,figsize=(10.6,2.55))
for ax,name,label in zip(axes,names,labels):
 row=z.loc[name];matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]],int);ax.imshow(np.log1p(matrix),cmap='Blues',vmin=0,vmax=np.log1p(1200))
 for y in range(2):
  for x in range(2):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',color='white' if matrix[y,x]>100 else '#162735',fontsize=11)
 ax.set(title=label,xticks=[0,1],xticklabels=['B','M'],yticks=[0,1],yticklabels=['B','M']);ax.set_xlabel('Predicted')
axes[0].set_ylabel('Actual');fig.tight_layout();save(fig,'04_confusion_matrices')

strat=pd.read_csv(r/'06_results/performance_by_source_and_tool.csv');rows=[]
for split,source in [('test','Domainator'),('test','DACA'),('test','GraphTunnel'),('external_test','Domainator'),('external_test','GraphTunnel')]:
 row=[]
 for name in names:
  part=strat[(strat.run_id==name)&(strat.split==split)&(strat.dimension=='source_name')&(strat.display_value==source)].iloc[0];row.append(part.recall*100)
 rows.append(row)
fig,ax=plt.subplots(figsize=(8.7,3.5));im=ax.imshow(rows,cmap='YlGnBu',vmin=70,vmax=100,aspect='auto')
for y,row in enumerate(rows):
 for x,v in enumerate(row):ax.text(x,y,f'{v:.2f}%',ha='center',va='center',color='white' if v>90 else '#142839')
ax.set(xticks=range(5),xticklabels=labels,yticks=range(5),yticklabels=['Test · Domainator','Test · DACA','Test · GraphTunnel','External · Domainator','External · GraphTunnel'])
ax.set_title('Malicious recall by capture ecosystem');fig.colorbar(im,ax=ax,label='Recall (%)',shrink=.8);fig.tight_layout();save(fig,'05_cross_source_recall')

v=pd.read_csv(r/'06_results/validation_sensitivity.csv');fig,axes=plt.subplots(1,4,figsize=(10.7,3.0))
for ax,kind in zip(axes,['rf','if','bilstm','sage']):
 part=v[v.model==kind];ax.bar(range(len(part)),part.fp,color=colors[['rf','if','bilstm','sage'].index(kind)])
 ax.set(title={'rf':'Random Forest','if':'Isolation Forest','bilstm':'BiLSTM','sage':'GraphSAGE'}[kind],xticks=range(len(part)),xticklabels=[x.replace(kind+'_','').replace('no_volume_timing','no volume + time').replace('samples1024','sample 1024') for x in part.run_id],ylim=(0,8));ax.tick_params(axis='x',labelrotation=65,labelsize=8)
axes[0].set_ylabel('Validation false positives');fig.suptitle('Sensitivity experiments · all candidates retained 100% Validation recall',fontsize=11);fig.tight_layout();save(fig,'06_validation_sensitivity')
print('Created six figure pairs')
