(function(){
  let calls = 100;
  let aht = 30; // seconds
  let agents = 10;
  const awt = 20; // target wait
  const slTarget = 0.8;

  function erlangB(traffic, agents){
    if(agents === 0) return 1.0;
    if(traffic === 0) return 0.0;
    let b = 1.0;
    for(let i=1;i<=agents;i++){
      b = (traffic*b)/(i + traffic*b);
    }
    return b;
  }

  function erlangC(traffic, agents){
    if(agents <= traffic) return 1.0;
    const eb = erlangB(traffic, agents);
    const rho = traffic/agents;
    if(rho >= 1) return 1.0;
    return eb / (1 - rho + rho*eb);
  }

  function serviceLevel(arrival, aht, agents, awt){
    const traffic = arrival*aht;
    if(agents <= traffic) return 0.0;
    const pc = erlangC(traffic, agents);
    if(pc === 0) return 1.0;
    const expFactor = Math.exp(-(agents - traffic)*awt/aht);
    return 1 - pc*expFactor;
  }

  function waitingTime(arrival, aht, agents){
    const traffic = arrival*aht;
    if(agents <= traffic) return Infinity;
    const pc = erlangC(traffic, agents);
    return (pc*aht)/(agents - traffic);
  }

  function occupancy(arrival, aht, agents){
    const traffic = arrival*aht;
    if(agents <= 0) return 1.0;
    return Math.min(traffic/agents, 1.0);
  }

  function requiredAgents(arrival, aht, awt, slTarget, maxAgents){
    for(let a=1;a<=maxAgents;a++){
      if(serviceLevel(arrival, aht, a, awt) >= slTarget) return a;
    }
    return maxAgents;
  }

  function renderService(sl){
    document.getElementById('service-level').textContent = (sl*100).toFixed(1)+'%';
  }

  function renderASA(asa){
    const bar = document.getElementById('asa-bar');
    const max = 120; // seconds
    const pct = Math.min(asa/max,1)*100;
    bar.style.width = pct+'%';
    bar.textContent = (asa === Infinity ? '∞' : asa.toFixed(1)+'s');
  }

  function renderBars(busy, available, waiting){
    const totalAgents = busy + available;
    const busyPct = totalAgents ? (busy/totalAgents)*100 : 0;
    const availPct = totalAgents ? (available/totalAgents)*100 : 0;
    const waitPct = (waiting/(waiting+totalAgents))*100;
    const busyBar = document.getElementById('busy-bar');
    const availBar = document.getElementById('available-bar');
    const waitBar = document.getElementById('waiting-bar');
    busyBar.style.width = busyPct+'%';
    busyBar.textContent = `${busy}`;
    availBar.style.width = availPct+'%';
    availBar.textContent = `${available}`;
    waitBar.style.width = waitPct+'%';
    waitBar.textContent = `${waiting}`;
  }

  function renderMatrix(busy, available, vacant){
    const container = document.getElementById('agents-matrix');
    container.innerHTML='';
    for(let i=0;i<busy;i++){
      const div=document.createElement('div');
      div.className='agent busy';
      container.appendChild(div);
    }
    for(let i=0;i<available;i++){
      const div=document.createElement('div');
      div.className='agent available';
      container.appendChild(div);
    }
    for(let i=0;i<vacant;i++){
      const div=document.createElement('div');
      div.className='agent vacant';
      container.appendChild(div);
    }
  }

  function renderQueue(len, asa){
    const container = document.getElementById('queue');
    container.innerHTML='';
    for(let i=0;i<len;i++){
      const span=document.createElement('span');
      span.className='queue-item';
      span.textContent='📞';
      const time=document.createElement('small');
      time.textContent=`${asa.toFixed(0)}s`;
      span.appendChild(time);
      container.appendChild(span);
    }
    if(len>0){
      const info=document.createElement('span');
      info.className='ms-2';
      info.textContent=`${len} en cola`;
      container.appendChild(info);
    }
  }

  function update(){
    const arrival = calls/3600;
    const traffic = arrival*aht;
    const sl = serviceLevel(arrival, aht, agents, awt);
    const asa = waitingTime(arrival, aht, agents);
    const occ = occupancy(arrival, aht, agents);
    const required = requiredAgents(arrival, aht, awt, slTarget, 200);
    const busy = Math.min(agents, Math.round(occ*agents));
    const available = Math.max(0, agents - busy);
    const vacant = Math.max(0, required - agents);
    const queueProb = erlangC(traffic, agents);
    const queued = Math.max(0, Math.round(queueProb*calls) - busy);
    renderService(sl);
    renderASA(asa);
    renderBars(busy, available, queued);
    renderMatrix(busy, available, vacant);
    renderQueue(queued, asa);
  }

  document.getElementById('btn-demand').addEventListener('click', function(){
    calls += Math.round(calls*0.1) || 1;
    update();
  });
  document.getElementById('btn-agents').addEventListener('click', function(){
    agents += 5;
    update();
  });
  document.getElementById('btn-aht').addEventListener('click', function(){
    aht = Math.max(5, aht-5);
    update();
  });

  update();
})();
