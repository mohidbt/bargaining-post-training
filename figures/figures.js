/* Shared pure SVG renderers for the offline gallery and static exports. */
const C={paper:'#f4ead2',ink:'#211a0e',muted:'#786a4d',rule:'#dcc8a0',seller:'#9c3520',buyer:'#2f527a',green:'#4f6b3f',gold:'#97742a',gray:'#786a4d'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const fmt=n=>Number(n).toFixed(3), pct=n=>`${(100*n).toFixed(1)}%`;
const txt=(x,y,s,size=15,attrs='')=>`<text x="${x}" y="${y}" font-size="${size}" ${attrs}>${esc(s)}</text>`;
const line=(x1,y1,x2,y2,color=C.rule,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${attrs}/>`;
const mark=(x,y,color,label,r=4,attrs='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" tabindex="0" role="img" aria-label="${esc(label)}" data-detail="${esc(label)}" ${attrs}><title>${esc(label)}</title></circle>`;
function wrap(title,body,h=400){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 ${h}" role="img" aria-label="${esc(title)}" style="background:${C.paper};color:${C.ink};font-family:Spectral,Georgia,serif"><title>${esc(title)}</title><rect width="900" height="${h}" fill="${C.paper}"/><g fill="${C.ink}">${body}</g></svg>`;}
function axis(x,y,w,h,xlo,xhi,ylo,yhi,xticks,yticks,xlabel){
 const X=v=>x+(v-xlo)/(xhi-xlo)*w,Y=v=>y+h-(v-ylo)/(yhi-ylo)*h;
 let s='';for(const v of yticks){s+=line(x,Y(v),x+w,Y(v));s+=txt(x-9,Y(v)+5,v,14,'text-anchor="end"');}for(const v of xticks){s+=line(X(v),y+h,X(v),y+h+5,C.muted);s+=txt(X(v),y+h+24,v,14,'text-anchor="middle"');}s+=txt(x+w/2,y+h+48,xlabel,16,'text-anchor="middle"');return {s,X,Y};
}
function series(points,a,color,label,dash=''){
 return `<polyline points="${points.map(([x,y])=>`${a.X(x)},${a.Y(y)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" ${dash?`stroke-dasharray="${dash}"`:''}/>`+points.map(([x,y])=>mark(a.X(x),a.Y(y),color,`${label}; step ${x}: ${fmt(y)}`,2.8)).join('');
}
function training(){
 let s=txt(65,35,'Training run two',25)+txt(65,62,'Full batches: reward and zero advantage. Archived pages: observed endings.',16);
 const roles=['seller','buyer'];const panels=[{y:95,title:'Mean midpoint reward',lo:-1,hi:1,ticks:[-1,-.5,0,.5,1],key:r=>`reward/wide-${r}/mean`},{y:385,title:'Zero advantage sample share',lo:0,hi:1,ticks:[0,.25,.5,.75,1],key:r=>`filters/wide-${r}/zero_advantage`}];
 for(const p of panels){s+=txt(65,p.y,p.title,19)+txt(650,p.y,'Seller',16,`fill="${C.seller}"`)+txt(745,p.y,'Buyer',16,`fill="${C.buyer}"`);const a=axis(65,p.y+18,760,175,0,51,p.lo,p.hi,[0,10,20,30,40,50],p.ticks,'Logged training step');s+=a.s;for(const role of roles){s+=series(DATA.curves.map(r=>[r.step,r[p.key(role)]]),a,C[role],`${role}, ${p.title}`);const v=DATA.curves.find(r=>r.step===50)[p.key(role)];s+=mark(a.X(50),a.Y(v),C[role],`${role}; logged step 50: ${fmt(v)}`,6,'stroke="#f4ead2" stroke-width="1"');}}
 s+=txt(65,685,'Observed ending shares in archived pages',19)+txt(505,685,'Deal',15,`fill="${C.green}"`)+txt(580,685,'Expiry',15,`fill="${C.gray}"`)+txt(660,685,'Walk',15,`fill="${C.gold}"`)+txt(740,685,'Forfeit',15,`fill="${C.seller}"`);const a=axis(65,704,760,165,0,51,0,1,[1,5,25,50],[0,.25,.5,.75,1],'Archived training step');s+=a.s;
 const cats=['deal','timeout','walk','forfeit'], colors=[C.green,C.gray,C.gold,C.seller];
 for(const [i,cat] of cats.entries()){const p=DATA.archived.map(r=>[r.step,(r.outcomes[cat]||0)/r.n]);s+=series(p,a,colors[i],`${cat}, archived n=64`,'5 4');}
 s+=txt(65,952,'Seller',16,`fill="${C.seller}"`)+txt(150,952,'Buyer',16,`fill="${C.buyer}"`)+txt(300,952,'Deal',16,`fill="${C.green}"`)+txt(375,952,'Expiry',16,`fill="${C.gray}"`)+txt(475,952,'Walk',16,`fill="${C.gold}"`)+txt(555,952,'Forfeit',16,`fill="${C.seller}"`);
 s+=txt(65,980,'Archive markers summarize 64 of 512 rollouts each; dashed segments connect observed pages.',15);
 return wrap('Reward, zero advantage and archived outcomes over training steps',s,1004);
}
function openings(steps=[5,50]){
 let s=txt(65,35,'Opening offers across private values',25);const colors={1:C.gold,5:C.buyer,25:C.green,50:C.seller};
 for(const [i,role] of ['seller','buyer'].entries()){
 const x=65+i*440;s+=txt(x,76,role==='seller'?'Seller: private cost c':'Buyer: private value v',20);
 const a=axis(x,100,340,230,role==='seller'?39:89,role==='seller'?61:111,0,150,role==='seller'?[40,50,60]:[90,100,110],[0,50,100,150],role==='seller'?'Private cost c':'Private value v');s+=a.s;
 for(const step of steps){const counts={};for(const row of DATA.openings.filter(r=>r.role===role&&r.step===step)){const key=`${row.own},${row.price}`;counts[key]=(counts[key]||0)+1;}
 for(const [key,n] of Object.entries(counts)){const [own,price]=key.split(',').map(Number);s+=mark(a.X(own),a.Y(price),colors[step],`${role}, step ${step}; private value ${own}, opening ${price}; ${n} archived rollouts`,3+Math.sqrt(n)*2,`fill-opacity="${step===50?'.8':'.4'}" stroke="${colors[step]}" stroke-width="1"`);}}
 }
 s+=txt(65,425,'Opening price',16)+steps.map((step,i)=>txt(220+i*130,425,`Step ${step}`,16,`fill="${colors[step]}"`)).join('');
 s+=txt(65,460,'Circle area reflects repeated rollouts at the same value and offer. No jitter is added.',15);
 s+=txt(65,489,'Step 50: seller 95 in 31/31 openings; buyer 55 in 33/33. Archive only.',16);
 let pos=518;for(const step of steps){const rows=DATA.openings.filter(r=>r.step===step);const exclusions=DATA.archived.find(r=>r.step===step).excluded_openings;s+=txt(65,pos,`Step ${step}: ${rows.length}/64 valid first-turn offers; excluded seller ${exclusions.seller||0}, buyer ${exclusions.buyer||0}.`,15);pos+=24;}
 return wrap('Opening price versus own reservation value in archived training rollouts',s,pos+20);
}
function endings(role='all'){
 let s=txt(65,35,'How the step-50 archive ended',25), a=axis(65,92,540,285,0,3,0,35,[],[0,5,10,15,20,25,30,35],'');s+=a.s;
 const cats=[['opponent positive','Opponent accepted, reward > 0',C.green],['opponent zero','Opponent accepted, reward = 0',C.gold],['opponent negative','Opponent accepted, reward < 0',C.seller],['learner positive','Learner accepted, reward > 0',C.buyer],['learner zero','Learner accepted, reward = 0','#a49269'],['learner negative','Learner accepted, reward < 0','#7a2616'],['timeout','Expiry',C.gray],['walk','Walk',C.gold],['forfeit','Forfeit',C.seller]];
 let totals=[];for(const [i,fam] of ['boulware','conceder','random_threshold'].entries()){const rows=DATA.endings.filter(e=>e.family===fam&&(role==='all'||e.role===role));totals.push(rows.length);let bottom=0;for(const [key,label,color] of cats){const n=rows.filter(e=>e.outcome==='deal'?key===`${e.actor} ${e.reward>0?'positive':e.reward<0?'negative':'zero'}`:key===e.outcome).length;if(!n)continue;const x=110+i*175,y=a.Y(bottom+n),h=n/35*285,detail=`${fam}, ${role}; ${label}: ${n} of ${rows.length}`;s+=`<rect x="${x}" y="${y}" width="90" height="${h}" fill="${color}" tabindex="0" role="img" aria-label="${esc(detail)}" data-detail="${esc(detail)}"><title>${esc(detail)}</title></rect>`;s+=txt(x+45,y+h/2+5,n,17,'fill="#f4ead2" text-anchor="middle"');bottom+=n;}s+=txt(155+i*175,402,fam==='random_threshold'?'TIOLI':fam==='boulware'?'Boulware':'Conceder',15,'text-anchor="middle"');s+=txt(155+i*175,424,`n=${rows.length}`,14,'text-anchor="middle"');}
 let y=100;for(const [key,label,color] of cats){const n=DATA.endings.filter(e=>(role==='all'||e.role===role)&&(e.outcome==='deal'?key===`${e.actor} ${e.reward>0?'positive':e.reward<0?'negative':'zero'}`:key===e.outcome)).length;if(!n)continue;s+=`<rect x="630" y="${y-12}" width="13" height="13" fill="${color}"/>`+txt(653,y,label,15);y+=34;}
 s+=txt(65,470,`${role==='all'?'Both roles':role}; ${totals.reduce((x,y)=>x+y,0)} archived rollouts. Deals show accepting actor and midpoint reward.`,16);
 s+=txt(65,500,'No agreed price exists for expiry, so no profit category is assigned to it.',15);
 return wrap('Step-50 archived rollout counts by opponent family and observed ending',s,530);
}
function bounds(role='seller'){
 let s=txt(65,35,'References against each opponent family',25)+txt(65,65,`${role}; references use identical refinement episodes, 8,000 total for this role.`,16);
 const a=axis(65,110,750,255,0,3,0,1,[],[0,.25,.5,.75,1],'');s+=a.s;
 for(const [i,fam] of ['boulware','conceder','random_threshold'].entries()){
 const lo=DATA.bounds.open_loop_winner.family[fam][role],hi=DATA.bounds.upper_full_information.family[fam][role],x=185+i*245;
 s+=line(x,a.Y(lo.mean),x,a.Y(hi.mean),C.rule,'stroke-width="12"');
 for(const [r,col,label] of [[lo,C.green,'Searched lower reference'],[hi,C.buyer,'Privileged full information upper bound']]){s+=mark(x,a.Y(r.mean),col,`${role}, ${fam}; ${label}: ${fmt(r.mean)}; n=${r.n}`,7);s+=txt(x+15,a.Y(r.mean)+5,fmt(r.mean),15,`fill="${col}"`);}
 s+=txt(x,393,fam==='random_threshold'?'TIOLI':fam==='boulware'?'Boulware':'Conceder',16,'text-anchor="middle"');s+=txt(x,416,`n=${lo.n}`,14,'text-anchor="middle"');}
 s+=txt(65,470,'Searched lower reference',16,`fill="${C.green}"`)+txt(390,470,'Privileged full information upper bound',16,`fill="${C.buyer}"`);
 s+=txt(65,501,'Mean midpoint reward. The upper bound uses hidden opponent parameters.',15);
 s+=txt(65,527,'The searched comparator is a history-blind offer sequence with standing-offer-reactive acceptance.',14);
 s+=txt(65,553,'It is a lower bound on what a policy with learner observations can achieve.',15);
 return wrap('Mean reward reference comparison by opponent family',s,580);
}
function frontier(scale='midpoint_reward'){
 const label=scale==='midpoint_reward'?'Midpoint reward':'Surplus share';let s=txt(65,35,'Frontier model on identical episodes',25)+txt(65,65,`F11 exploratory evaluation; 64 episodes for each role. ${label}.`,16);
 const a=axis(65,110,750,220,0,.7,0,2,[0,.1,.2,.3,.4,.5,.6,.7],[],`Mean ${label.toLowerCase()}`);s+=a.s;
 for(const [i,role] of ['seller','buyer'].entries()){
 const l=DATA.frontier.lower[role][scale],u=DATA.frontier.upper[role][scale],y=155+i*110;s+=txt(65,y-23,role,18);
 const rows=[[l.model_mean,C.seller,'Frontier model'],[l.blind_mean,C.green,'Searched comparator'],[u.blind_mean,C.buyer,'Full information upper bound']];
 s+=line(a.X(l.blind_mean),y,a.X(u.blind_mean),y,C.rule,'stroke-width="8"');
 for(const [value,color,name] of rows){s+=mark(a.X(value),y,color,`${role}; ${name}: ${fmt(value)}`,7);s+=txt(a.X(value),y+30,fmt(value),15,'text-anchor="middle"');}
 }
 s+=txt(65,415,'Frontier model',16,`fill="${C.seller}"`)+txt(230,415,'Searched comparator',16,`fill="${C.green}"`)+txt(485,415,'Full information upper bound',16,`fill="${C.buyer}"`);
 s+=txt(65,455,'Model minus searched comparator, bootstrap 95% intervals:',17);
 for(const [i,role] of ['seller','buyer'].entries()){const r=DATA.frontier.lower[role][scale];s+=txt(65,488+i*28,`${role}: ${fmt(r.paired_mean_model_minus_blind)} [${fmt(r.ci95[0])}, ${fmt(r.ci95[1])}]`,16);}
 s+=txt(65,551,'References are replayed on the model’s own episodes. Differences do not measure opponent inference.',14);
 if(scale==='surplus_share')s+=txt(65,577,'Surplus share is zero when no deal occurs; the reward-to-share conversion applies only to deals.',14);
 return wrap(`Frontier model and references on identical episodes, ${label}`,s,scale==='surplus_share'?605:580);
}
function mechanism(p=70,c=50,v=100){
 const reward=Math.max(-1,Math.min(1,(2*p-c-v)/(v-c))),seller=p-c,buyer=v-p;
 let s=txt(65,35,'One price splits the surplus',25)+txt(65,65,`Seller cost c=${c}, buyer value v=${v}. Move the agreed price p.`,17);
 const a=axis(65,110,750,120,40,110,-1,1,[40,50,60,70,80,90,100,110],[-1,0,1],'Agreed price p');s+=a.s;
 for(const [role,sign] of [['seller',1],['buyer',-1]]){const pts=Array.from({length:71},(_,i)=>[40+i,sign*Math.max(-1,Math.min(1,(2*(40+i)-c-v)/(v-c)))]);s+=series(pts,a,C[role],`${role} midpoint reward`);s+=mark(a.X(p),a.Y(sign*reward),C[role],`${role}; price ${p}, reward ${fmt(sign*reward)}`,7);}
 s+=line(a.X((c+v)/2),110,a.X((c+v)/2),230,C.gold,'stroke-dasharray="4 4"');
 s+=txt(65,330,`Seller surplus p−c = ${seller}`,22,`fill="${C.seller}"`)+txt(480,330,`Buyer surplus v−p = ${buyer}`,22,`fill="${C.buyer}"`);
 s+=txt(65,365,`Seller reward ${fmt(reward)}`,22,`fill="${C.seller}"`)+txt(480,365,`Buyer reward ${fmt(-reward)}`,22,`fill="${C.buyer}"`);
 s+=txt(65,413,`Total surplus v−c = ${v-c}. Midpoint = ${(c+v)/2}.`,18);
 s+=txt(65,446,'Surplus can be positive while midpoint reward is negative. Walking and expiry return zero.',15);
 s+=txt(65,474,'Reward = clip((2p−c−v)/(v−c), −1, 1) for seller; buyer gets its negative.',15);
 return wrap('Illustration of economic surplus and midpoint rewards at an agreed price',s,500);
}
function concessions(){
 let s=txt(65,35,'Three sequences of offers',25)+txt(65,65,'Illustration: seller opens at 110, cost 50, deadline at round 6.',17);
 const a=axis(65,115,750,260,1,6,40,120,[1,2,3,4,5,6],[40,60,80,100,120],'Action round k');s+=a.s;
 for(const [name,e,color] of [['Boulware',.2,C.seller],['Conceder',3,C.buyer],['TIOLI',null,C.green]]){const points=Array.from({length:6},(_,i)=>[i+1,e===null?85:Math.round(110+(50-110)*Math.pow(i/5,1/e))]);s+=series(points,a,color,name);}
 s+=txt(65,453,'Boulware e=0.2',16,`fill="${C.seller}"`)+txt(300,453,'Conceder e=3',16,`fill="${C.buyer}"`)+txt(535,453,'TIOLI threshold=85',16,`fill="${C.green}"`);
 s+=txt(65,493,'Targets are integer prices: round(110 + (50−110) × ((k−1)/5)^(1/e)).',16);
 s+=txt(65,523,'A bot accepts a standing offer at least as good as its next target.',15);
 s+=txt(65,550,'Time bots also accept at their reservation at the deadline. These curves are examples, not data.',15);
 return wrap('Illustrative opponent concession curves with the implemented offer rule',s,580);
}
function ranges(cmin=40,cmax=60,vmin=90,vmax=110){
 const min=(cmin+vmin)/2,max=(cmax+vmax)/2,mean=(min+max)/2;let s=txt(65,35,'Where could the midpoint be?',25)+txt(65,65,'Explore independent uniform ranges. Fixed opening offers stay at seller 95 and buyer 55.',16);
 const low=Math.min(cmin,vmin,55)-10,high=Math.max(cmax,vmax,95)+10;
 const a=axis(65,105,750,260,low,high,0,4,[],[],'Price');s+=a.s;
 const X=a.X;for(const [name,lo,hi,y,color] of [['Seller cost',cmin,cmax,140,C.seller],['Buyer value',vmin,vmax,205,C.buyer],['Possible midpoint',min,max,270,C.gold]]){s+=line(X(lo),y,X(hi),y,color,'stroke-width="16" stroke-linecap="round"');s+=txt(65,y-20,`${name}: ${lo} to ${hi}`,16);s+=mark(X(lo),y,color,`${name} minimum ${lo}`,5);s+=mark(X(hi),y,color,`${name} maximum ${hi}`,5);}
 s+=line(X(55),315,X(95),315,C.rule,'stroke-width="4"')+mark(X(55),315,C.buyer,'Fixed buyer opening: 55',8)+mark(X(95),315,C.seller,'Fixed seller opening: 95',8);s+=txt(X(55),347,'Buyer 55',15,'text-anchor="middle"')+txt(X(95),347,'Seller 95',15,'text-anchor="middle"');
 s+=txt(65,435,`Midpoint range ${min} to ${max}; expected midpoint ${mean}.`,20);
 s+=txt(65,471,`Minimum available surplus v_min−c_max = ${vmin-cmax}.`,18);
 s+=txt(65,507,'This changes the possible midpoint, not a measured training outcome or an optimal opening.',15);
 s+=txt(65,536,'Positive reward at every possible midpoint: seller p > maximum midpoint; buyer p < minimum.',15);
 return wrap('Illustrative midpoint range from cost and value ranges with fixed openings',s,565);
}
const renderers={training,openings,endings,bounds,frontier,mechanism,concessions,ranges};
if(typeof module!=='undefined')module.exports={renderers};
