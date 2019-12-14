# DataAnalysis
2019 가을학기 데이터문제해결및실습1 

* Recommender System - LastFM data
## Final Assignment
1. 과제 모듈화
2. Singularity와 PIP 구현


### PIP (Proximity-Impact-Popularity)
두 사용자 <a href="https://www.codecogs.com/eqnedit.php?latex=u_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{i}" title="u_{i}" /></a> 와 <a href="https://www.codecogs.com/eqnedit.php?latex=u_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{j}" title="u_{j}" /></a>의 유사도 측정 : 


<a href="https://www.codecogs.com/eqnedit.php?latex=SIM(u_{i},u_{j})&space;=&space;\sum_{k\in&space;C_{i,j}}PIP(r_{ik},r_{jk})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SIM(u_{i},u_{j})&space;=&space;\sum_{k\in&space;C_{i,j}}PIP(r_{ik},r_{jk})" title="SIM(u_{i},u_{j}) = \sum_{k\in C_{i,j}}PIP(r_{ik},r_{jk})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=PIP(r1,&space;r2)&space;=&space;Proximity(r1,&space;r2)&space;\times&space;Impact(r1,&space;r2)&space;\times&space;Popularity(r1,&space;r2)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?PIP(r1,&space;r2)&space;=&space;Proximity(r1,&space;r2)&space;\times&space;Impact(r1,&space;r2)&space;\times&space;Popularity(r1,&space;r2)." title="PIP(r1, r2) = Proximity(r1, r2) \times Impact(r1, r2) \times Popularity(r1, r2)." /></a>
