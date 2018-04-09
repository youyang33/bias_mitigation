var dropSize = 80, dropR = 20, axisOptions = {};
var HOVER_WEIGHT = 1, DRAG_WEIGHT = 2, CLICK_WEIGHT = 3, DOUBLE_CLICK_WEIGHT = 4; 
var mouseoverStart = 0, mouseoverEnd = 0, dragStart = 0, dragEnd = 0; 
var isDragging = false; // used to make sure that hovers while dragging don't get logged
var biasOpacityScale = d3.scaleLinear()
	.domain([0.0, 1.0])
	.range([0.1, 1.0]);
var biasWidthScale = d3.scaleLinear()
	.domain([0.0, 1.0])
	.range([5, 150]);
var biasResults = [0, 0, 0, 0, 0, 0];


registerKeyboardHandler = function(callback) {
  var callback = callback;
  d3.select(window).on("keydown", callback);  
};

SimpleGraph = function(elemid, data, options) {
  var self = this;
  this.chart = document.getElementById(elemid);
  this.cx = this.chart.clientWidth;
  this.cy = this.chart.clientHeight - 125;
  this.options = options || {};
  this.points = data;
  
  if (istxtdata==true) {
    var cents = d3.nest()
      .key(function(d) {return d["label"];}) 
      .rollup(function(d) {  
       return {"x": d3.mean(d, function(g) {return g["x"]; }), "y": d3.mean(d, function(g) {return g["y"]; })};
      }).entries(data);

    var topic = {"1": "tree,node,hierarchi,structur,space","2": "user,system,interfac,present,result","3": "layout,algorithm,treemap,space,hierarch","4": "techniqu,interact,displai,explor,larg","5": "network,node,social,link,explor","6": "graph,cluster,node,edg,draw","7": "analysi,system,analyt,tool,design","8": "data,set,analysi,larg,pattern","9": "inform,space,displai,document,context","10": "visual,design,base,model,paper"};
    
    cents.forEach(function(d,i) {
      d["values"]["topic"] = topic[d["key"]];
      data.forEach(function(p,j) {
        if (d["key"]==p["label"]) {
          var shrink = 0.85;
          if (self.options.init=="X" || self.options.init==true) { p["x"] = (1-shrink) * d["values"]["x"] + shrink * p["x"]; }
          if (self.options.init=="Y" || self.options.init==true) { p["y"] = (1-shrink) * d["values"]["y"] + shrink * p["y"]; }
        }
      });
    });

    this.cents = cents;
  }

  this.options.xmax = d3.max(data, function(d) { return +d["x"]; }) || options.xmax || 30;
  this.options.xmin = d3.min(data, function(d) { return +d["x"]; }) || options.xmin || 0;
  this.options.ymax = d3.max(data, function(d) { return +d["y"]; }) || options.ymax || 10;
  this.options.ymin = d3.min(data, function(d) { return +d["y"]; }) || options.ymin || 0;
  
  // copy these original values to variables to preserve
  this.options.origxmax = this.options.xmax;
  this.options.origxmin = this.options.xmin;
  this.options.origymax = this.options.ymax;
  this.options.origymin = this.options.ymin;
  
  // re-scale the max and min values to account for jittering by 5% on either side
  this.options.xmax = this.options.xmax + 0.05 * Math.abs(this.options.xmax - this.options.xmin);
  this.options.xmin = this.options.xmin - 0.05 * Math.abs(this.options.xmax - this.options.xmin);
  this.options.ymax = this.options.ymax + 0.05 * Math.abs(this.options.ymax - this.options.ymin);
  this.options.ymin = this.options.ymin - 0.05 * Math.abs(this.options.ymax - this.options.ymin);
  
  this.padding = {
     "top":    this.options.title  ? 40 : 20,
     "right":                 			      30,
     "bottom": this.options.xlabel ? 230 : 10,
     "left":   150
  };

  this.size = {
    "width":  this.cx - this.padding.left - this.padding.right,
    "height": this.cy - this.padding.top  - this.padding.bottom
  };

  axisOptions = {
    "X": { "width": this.size.width-dropSize*2,
          "height": this.padding.bottom-40,
          "padding": {top: this.padding.top+this.size.height+40, right: 0, left: this.padding.left+dropSize+10, bottom: 0} },
    "Y": { "width": this.padding.left-dropSize,
          "height": this.size.height-dropSize*2,
          "padding": {top: this.padding.top+dropSize, right: 0, left: 15, bottom: 0} }
  };
  
  // x-scale
  this.x = d3.scale.linear()
      .domain([this.options.xmin, this.options.xmax])
      .range([0, this.size.width]);

  // drag x-axis logic
  this.downx = Math.NaN;

  // y-scale (inverted domain)
  this.y = d3.scale.linear()
      .domain([this.options.ymax, this.options.ymin])
      .nice()
      .range([0, this.size.height])
      .nice();

  // drag y-axis logic
  this.downy = Math.NaN;

  this.dragged = this.selected = null;
  this.dropped = null;
  if (this.options.dropzone) {this.dropzone = this.options.dropzone; delete this.options.dropzone;}
  else {this.dropzone = {"YH":[], "YL":[], "XL":[], "XH":[]};}

  var xrange =  (this.options.xmax - this.options.xmin),
      yrange2 = (this.options.ymax - this.options.ymin) / 2,
      yrange4 = yrange2 / 2;

  if (this.options.init==true) {
    var SC = d3.select(this.chart).append("svg")
      .attr("width",  this.cx)
      .attr("height", this.cy);

    var drp = SC.append("g").attr("id", "DROP");
    var xlGroup = drp.append("g").attr("id", "XL");
    xlGroup.append("rect").attr("x", this.padding.left).attr("y", this.cy-this.padding.bottom+dropSize*0.5+20);
    xlGroup.append("text").attr("x", this.padding.left+dropSize/2.0).attr("y", this.cy-this.padding.bottom+dropSize*2).attr("text-anchor", "middle").attr("font-size", "14px").text("'A' Exemplars");
    var xhGroup = drp.append("g").attr("id", "XH");
    xhGroup.append("rect").attr("x", this.cx-this.padding.right-dropSize).attr("y", this.cy-this.padding.bottom+dropSize*0.5+20);
    xhGroup.append("text").attr("x", this.cx-this.padding.right-dropSize/2.0).attr("y", this.cy-this.padding.bottom+dropSize*2).attr("text-anchor", "middle").attr("font-size", "14px").text("'B' Exemplars");
    drp.selectAll("rect").attr("width", dropSize).attr("height", dropSize).attr("rx", dropR).attr("ry", dropR);

    div = document.getElementById("btnXc");
    div.style.left = this.size.width/2+dropSize+175+10;
    div.style.top = this.padding.top+this.size.height+60;

    d3.select("#cbY")
      .selectAll("option")
      .data(attr)
      .enter()
      .append("option")
      .attr("value", function(d) {return d;})
      .text(function(d) {return d;});

    d3.select("#cbX")
      .selectAll("option")
      .data(attr)
      .enter()
      .append("option")
      .attr("value", function(d) {return d;})
      .text(function(d) {return d;});

    div = document.getElementById("cbY");
    div.style.left = this.padding.left-dropSize-100/2;
    div.style.top = this.padding.top+this.size.height/2;
    div = document.getElementById("cbX");
    div.style.left = this.size.width/2+dropSize;
    div.style.top = this.padding.top+this.size.height+60;
  }

  else {
    var SC = d3.select(this.chart).select("svg")
      .attr("width",  this.cx)
      .attr("height", this.cy);
  }

  this.vis = SC.append("g")
        .attr("id", "SC")
        .attr("transform", "translate(" + this.padding.left + "," + this.padding.top + ")");

  this.plot = this.vis.append("rect")
      .attr("width", this.size.width)
      .attr("height", this.size.height)
      .style("fill", "#EEEEEE")
      .attr("pointer-events", "all")
      this.plot.call(d3.behavior.zoom().x(this.x).y(this.y).on("zoom", this.redraw()));

  this.vis.append("svg")
      .attr("top", 0)
      .attr("left", 0)
      .attr("width", this.size.width)
      .attr("height", this.size.height)
      .attr("viewBox", "0 0 "+(this.size.width)+" "+(this.size.height));

  // add Chart Title
  if (this.options.title) {
    this.vis.append("text")
        .attr("class", "axis")
        .text(this.options.title)
        .attr("x", this.size.width/2)
        .attr("dy","-0.8em")
        .style("text-anchor","middle");
  }

  d3.select(this.chart)
      .on("mousemove.drag", self.mousemove())
      .on("touchmove.drag", self.mousemove())
      .on("mouseup.drag",  self.mouseup())
      .on("touchend.drag",  self.mouseup());

  this.redraw()();
};
  
//
// SimpleGraph methods
//

SimpleGraph.prototype.plot_drag = function() {
  var self = this;
  return function() {
    registerKeyboardHandler(self.keydown());
    d3.select('body').style("cursor", "move");
    if (d3.event.altKey) {
      var p = d3.mouse(self.vis.node());
      var newpoint = {};
      newpoint.x = self.x.invert(Math.max(0, Math.min(self.size.width,  p[0])));
      newpoint.y = self.y.invert(Math.max(0, Math.min(self.size.height, p[1])));
      self.points.push(newpoint);
      self.points.sort(function(a, b) {
        if (a.x < b.x) { return -1 };
        if (a.x > b.x) { return  1 };
        return 0
      });
      self.selected = newpoint;
      self.update();
      d3.event.preventDefault();
      d3.event.stopPropagation();
    }    
  }
};

SimpleGraph.prototype.update = function() {
  var self = this;

  var circle = this.vis.select("svg").selectAll("circle")
      .data(this.points);
  
  circle.enter().append("circle")
      .attr("class", function(d) {
    	  if (d.indropzone && d["coord"]["userlabel"] != label.DEFAULT) return "dropped classified";
    	  else if (d.indropzone) return "dropped";
    	  else if (d["coord"]["userlabel"] != label.DEFAULT) return "classified";
    	  else return null;
       }).attr("cx",    function(d) { return self.x(d["x"], 'x'); })
      .attr("cy",    function(d) { return self.y(d["y"], 'y'); })
      .attr("r", 7.0)
      .style("cursor", "resize")
      .style("fill", function(d) { return colorMap[d["coord"]["userlabel"]]; })
      .style("fill-opacity", function(d) { if (d["coord"]["userlabel"] != label.DEFAULT) return 1.0; else return 0.2; })
      .on("mouseover", function(d) {
        if (!isDragging) 
          mouseoverStart = new Date();
        tabulate(d);
        d3.select("#DROP").selectAll("circle").filter(function(c) {return c==d;}).attr("class", "highlighted");
      })
      .on("mouseout", function(d) {
    	  d3.select("#DROP").selectAll("circle").filter(function(c) {return c==d;}).classed("highlighted", false).classed("classified", function(c) {if (c["coord"]["userlabel"] == "Un-Assign") return false; else return true;});
        
        if (!isDragging) {
          mouseoverEnd = new Date();
          // units are seconds -- mouseover time will always encapsulate drag time as well
          var elapsedTime = (mouseoverEnd - mouseoverStart) / 1000;
          // log the mouseover with IAL
          ial.usermodel.incrementItemWeight(d, HOVER_WEIGHT, true, {'level': 'INFO', 'eventType': 'hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': 'scatterplot'});
          LE.log(JSON.stringify(ial.logging.peek()));
          //updateBias(false);
        }
        
        tabulate(d, 'empty');

      }).on("click", function(d){
    	  if (activePosition != "none") {
    		  d["coord"]["userlabel"] = activePosition;
    		  d3.select(this).style("fill", colorMap[activePosition]);
    		  if (activePosition != 'Un-Assign') d3.select(this).style("fill-opacity", 1.0);
    		  else d3.select(this).style("fill-opacity", 0.2);
    		  playerPositionMap[d.Name] = activePosition;
    		  ial.usermodel.incrementItemWeight(d, CLICK_WEIGHT, true, {'level': 'INFO', 'eventType': 'click', 'classification': activePosition, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': 'scatterplot'});
    		  LE.log(JSON.stringify(ial.logging.peek()));
    	  } else {
    		  ial.usermodel.incrementItemWeight(d, CLICK_WEIGHT, true, {'level': 'INFO', 'eventType': 'click', 'classification': 'none', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': 'scatterplot'});
    		  LE.log(JSON.stringify(ial.logging.peek()));
    	  }
    	  tabulate(d,"click"); 
    	  sglclick(d);
    	  d3.select("#DROP").selectAll("circle")
    	  	.each(function(e) {
    	  		if (d == e) {
    	  			d3.select(this).style("fill", colorMap[activePosition]);
    	  			d3.select(this).style("fill-opacity", 1.0);
    	  		}
    	  	});
      }).on("mousedown.dragstart", function(d) {
        isDragging = true; 
        dragStart = new Date();
      }).on("mousedown.drag", self.datapoint_drag()) 
      .on("touchstart.drag", self.datapoint_drag())
      .on("mouseup.dragend", function(d) { 
        if (isDragging) {
          dragEnd = new Date(); 
          // units are seconds
          var elapsedTime = (dragEnd - dragStart) / 1000;
          // log the drag with IAL
          ial.usermodel.incrementItemWeight(d, DRAG_WEIGHT, true, {'level': 'INFO', 'eventType': 'drag', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': 'scatterplot'});
          LE.log(JSON.stringify(ial.logging.peek()));
          //updateBias(false);
          isDragging = false; 
        }
      });
  
  if (istxtdata==true) {
    circle.attr("text", function(d) { return d["text"]["title"]; })
      .style("fill", function(d) { return color(d["label"]); })
      .style("stroke", function(d) { return color(d["label"]); });
  }
  else {circle.attr("text", function(d) { return d["Name"]; })}

  circle
      .attr("class", function(d) { return d.indropzone ? "dropped" : null; })
      .attr("cx",    function(d) { return self.x(d.x); })
      .attr("cy",    function(d) { return self.y(d.y); });

  circle.exit().remove();

  if (istxtdata==true) {
    var rect = this.vis.select("svg").selectAll("rect")
        .data(this.cents);

    rect.enter().append("rect")
        .attr("x",    function(d) { return self.x(d["values"].x)-10; })
        .attr("y",    function(d) { return self.y(d["values"].y)-10; });

    var textlabel = this.vis.select("svg").selectAll("text")
        .data(this.cents);

    textlabel.enter().append("text")
        .attr("id",    function(d) { return 'topic'+d["key"]; })
        .attr("x",    function(d) { return self.x(d["values"].x); })
        .attr("y",    function(d) { return self.y(d["values"].y); })
        // .attr("dy", "-1.5em")
        .attr("text-anchor", "middle")
        .text(function(d) {return d["values"]["topic"]})
        .style("fill", function(d) { return color(d["key"]); })
        .style("font-weight", "bold")
        .style("font-size", 14)
        .style("font-fanmily", "Arial");

    textlabel.attr("x",    function(d) { return self.x(d["values"].x); })
        .attr("y",    function(d) { return self.y(d["values"].y); })
        // .attr("dy", "-1.5em")
        .attr("text-anchor", "middle")
        .text(function(d) {return d["values"]["topic"];})
        .style("fill", function(d) { return color(d["key"]); })
        .style("font-weight", "bold")
        .style("font-size", 14)
        .style("font-fanmily", "Arial");

    textlabel.exit().remove();

    rect.attr("x",    function(d) { return self.x(d["values"].x)-document.getElementById("topic"+d["key"]).getBBox().width/2; })
        .attr("y",    function(d) { return self.y(d["values"].y)-12; })
        // .attr("dy", "-1.5em")
        .attr("width",    function(d) { return document.getElementById("topic"+d["key"]).getBBox().width; })
        .attr("height",   16)
        .style("fill", '#efefef')
        .style("opacity", 0.7);
        // .style("stroke-width", 3)
        // .style("stroke", function(d) { return color(d["key"]); });

    rect.exit().remove();
  }

  if (d3.event && d3.event.keyCode) {
    d3.event.preventDefault();
    d3.event.stopPropagation();
  }
}

sglclick = function(d) {
  if (x0!=d) {
    x0old = x0;
    x0 = d;
  }
};

dblclick = function(d) {

};

SimpleGraph.prototype.datapoint_drag = function() {
  var self = this;
  return function(d) {
    registerKeyboardHandler(self.keydown());
    document.onselectstart = function() { return false; };
    self.selected = self.dragged = d;
    self.dragged.oldy = d.y;
    self.dragged.oldx = d.x;

    

    self.update();
  }
};

SimpleGraph.prototype.mousemove = function() {
  var self = this; 
  return function() {
    var p = d3.mouse(self.vis[0][0]),
        t = d3.event.changedTouches;

    if (self.dragged) {
      self.dragged.y = self.y.invert(p[1]);
      self.dragged.x = self.x.invert(p[0]);
 
      if (0<=p[0] && p[0]<=dropSize && self.size.height+33+$("#instructions").height()<=p[1] && p[1]<=self.size.height+dropSize+33+$("#instructions").height()) { // XL
        self.dropped = "XL";
      } else if (self.size.width-dropSize<=p[0] && p[0]<=self.size.width && self.size.height+33+$("#instructions").height()<=p[1] && p[1]<=self.size.height+dropSize+33+$("#instructions").height()) { // XH
        self.dropped = "XH";
      }
      else {
        self.dropped = null;
      }
      
      self.update();
    };
    if (!isNaN(self.downx)) {
      d3.select('body').style("cursor", "ew-resize");
      var rupx = self.x.invert(p[0]),
          xaxis1 = self.x.domain()[0],
          xaxis2 = self.x.domain()[1],
          xextent = xaxis2 - xaxis1;
      if (rupx != 0) {
        var changex, new_domain;
        changex = self.downx / rupx;
        new_domain = [xaxis1, xaxis1 + (xextent * changex)];
        self.x.domain(new_domain);
        self.redraw()();
      }
      d3.event.preventDefault();
      d3.event.stopPropagation();
    };
    if (!isNaN(self.downy)) {
      d3.select('body').style("cursor", "ns-resize");
      var rupy = self.y.invert(p[1]),
          yaxis1 = self.y.domain()[1],
          yaxis2 = self.y.domain()[0],
          yextent = yaxis2 - yaxis1;
      if (rupy != 0) {
        var changey, new_domain;
        changey = self.downy / rupy;
        new_domain = [yaxis1 + (yextent * changey), yaxis1];
        self.y.domain(new_domain);
        self.redraw()();
      }
      d3.event.preventDefault();
      d3.event.stopPropagation();
    }
  }
};

SimpleGraph.prototype.mouseup = function() {
  var self = this;

  return function() {
	if ((self.dropped == "XH" && self.dropzone["XL"].indexOf(self.dragged) > -1) || (self.dropped == "XL" && self.dropzone["XH"].indexOf(self.dragged) > -1)) {
		self.dragged.y = self.dragged.oldy;
	    self.dragged.x = self.dragged.oldx;
	    self.dragged = null;
	    self.dropped = null;
	    self.redraw()();
    	self.downx = Math.NaN;
        self.downy = Math.NaN;
        d3.event.preventDefault();
        d3.event.stopPropagation();
        alert("Warning: A data point may not be dragged to both bins at the same time.");
    	return;
    }
    document.onselectstart = function() { return true; };
    d3.select('body').style("cursor", "auto");
    d3.select('body').style("cursor", "auto");
    if (!isNaN(self.downx)) {
      self.redraw()();
      self.downx = Math.NaN;
      d3.event.preventDefault();
      d3.event.stopPropagation();
    };
    if (!isNaN(self.downy)) {
      self.redraw()();
      self.downy = Math.NaN;
      d3.event.preventDefault();
      d3.event.stopPropagation();
    }
    if (self.dragged) { 
      self.dragged.y = self.dragged.oldy;
      self.dragged.x = self.dragged.oldx;

      if (self.dropped) {
        self.dragged.indropzone = true;
        var count = 0;
        for (var i = 0; i<self.dropzone[self.dropped].length; i++) {
          if (self.dropzone[self.dropped][i]["Name"]==self.dragged["Name"]) {
            count = count + 1;
            break;
          }
        }
        if (count==0) {
          self.dropzone[self.dropped][self.dropzone[self.dropped].length] = self.dragged;
          d3.select("#"+self.dropped).selectAll("circle").remove();

          var cx = +d3.select("#"+self.dropped).select("rect").attr("x")+0.5*dropSize,
              cy = +d3.select("#"+self.dropped).select("rect").attr("y")+0.5*dropSize,
              num = self.dropzone[self.dropped].length,
              dist = num==1 ? 0 : 10.0/Math.sin(Math.PI/num);
          
          d3.select("#"+self.dropped).selectAll("circle").data(self.dropzone[self.dropped]).enter().append("circle")
            .attr("cx",    function(d,i) { return cx + dist*Math.cos(Math.PI*2*i/num); })
            .attr("cy",    function(d,i) { return cy + dist*Math.sin(Math.PI*2*i/num); })
            .attr("r", 7.0)
            .style("fill", function(d) {
            	return colorMap[d["coord"]["userlabel"]];
            })
            .classed("classified", function(d) {
            	if (d["coord"]["userlabel"] != label.DEFAULT) return true;
                	else return false;
            })
            .on("mouseover", function(d) {
              tabulate(d); 
              tmpclass = d3.select("#SC").selectAll("circle").filter(function(c) {return c==d;}).attr("class");
              d3.select("#SC").selectAll("circle").filter(function(c) {return c==d;}).attr("class", "highlighted");

              if (!isDragging) 
                  mouseoverStart = new Date();
            })
            .on("mouseout", function(d) {
              d3.select("#SC").selectAll("circle").filter(function(c) {return c==d;}).attr("class", tmpclass);
              
              if (!isDragging && this.parentNode != null) { // make sure it hasn't been removed from the bin since mouseover began
            	  mouseoverEnd = new Date();
                  // units are seconds -- mouseover time will always encapsulate drag time as well
                  var elapsedTime = (mouseoverEnd - mouseoverStart) / 1000;
                  // log the mouseover with IAL
                  var whichBin = "none";
                  if (this.parentNode.id == "XL")
                	  whichBin = "XL"
                  else if (this.parentNode.id == "XH")
                	  whichBin = "XH";
                  ial.usermodel.incrementItemWeight(d, HOVER_WEIGHT, true, {'level': 'INFO', 'eventType': 'hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': whichBin});
                  LE.log(JSON.stringify(ial.logging.peek()));
              }
              tabulate(d, 'empty');
            })
            .on("dblclick", function(d) {
              var whichBin = "none";
              if (this.parentNode.id == "XL")
            	  whichBin = "XL"
              else if (this.parentNode.id == "XH")
            	  whichBin = "XH";
              ial.usermodel.incrementItemWeight(d, DOUBLE_CLICK_WEIGHT, true, {'level': 'INFO', 'eventType': 'double_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'pointLocation': whichBin});
              LE.log(JSON.stringify(ial.logging.peek()));
              
              thisdropzone = d3.select(this.parentNode).attr("id");
              for (var i = 0; i<self.dropzone[thisdropzone].length; i++) {
                if (self.dropzone[thisdropzone][i]==d) {
                  self.dropzone[thisdropzone].splice(i,1);
                  break;
                }
              }
              var inotherdropzone = graph.dropzone.XH.concat(graph.dropzone.XL,graph.dropzone.YH,graph.dropzone.YL)
                                    .filter(function(c) {return c==d;});
              if (inotherdropzone.length==0) {d.indropzone = false; tmpclass = null;}
              this.remove();
              console.log(self.dropzone);
              if ((thisdropzone=="XL" || thisdropzone=="XH") && self.dropzone["XH"].length*self.dropzone["XL"].length>0) {
            	  console.log("update X");
            	  updategraph("X");
              } else if ((thisdropzone=="YL" || thisdropzone=="YH") && self.dropzone["YH"].length*self.dropzone["YL"].length>0) {
                console.log("update Y");
                updategraph("Y");
              } else {
                self.redraw()();
              }
            })
            .on("click", function(d) {tabulate(d,"click");});
        }
      }
      if (self.dropped=="XL" || self.dropped=="XH") {
        if (isDragging) {
          dragEnd = new Date(); 
          // units are seconds
          var elapsedTime = (dragEnd - dragStart) / 1000;
          // log the drag with IAL
          ial.usermodel.incrementItemWeight(self.dragged, DRAG_WEIGHT, true, {'level': 'INFO', 'eventType': 'drag', 'bin': self.dropped, 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
          LE.log(JSON.stringify(ial.logging.peek()));
          //updateBias(false);
          isDragging = false; 
        }
      }
      if ((self.dropped=="XL" && self.dropzone["XH"].length>0) || (self.dropped=="XH" && self.dropzone["XL"].length>0)) {
        console.log("update X");
        updategraph("X");
      }
      self.dragged = null;
      self.dropped = null;
      self.redraw()();
    }
  }
}

updategraph = function(axistobeupdated,givenV,givenVchanged) {
  var exists = false;
  $('#cbX option').each(function(){
    if (this.value == "Custom Axis") {
      exists = true;
      return false;
    }
  });
  if (exists == false) {
    d3.select("#cbX").append("option").attr("value", "Custom Axis").text("Custom Axis").attr("id", "customAxisOption");
  }
  $("#cbX").val("Custom Axis");
  
  data = graph.points;
  if (givenV == undefined) {
    var x1 = {}, x0 = {};
    var high = graph.dropzone[axistobeupdated+"H"], low = graph.dropzone[axistobeupdated+"L"];
    for (var i = 0; i<attrNo; i++) {
      x1[attr[i]] = d3.mean(low, function(d) { return d["coord"][attr[i]]});
      x0[attr[i]] = d3.mean(high, function(d) { return d["coord"][attr[i]]});
    }

    var hlpair = [];
    for (var i = 0; i<high.length; i++) {
      for (var j = 0; j<low.length; j++) {
        var tmpelt = {};
        for (var k = 0; k<attrNo; k++) {
          tmpelt[attr[k]] = high[i]["coord"][attr[k]] - low[j]["coord"][attr[k]];
        }
        hlpair[hlpair.length] = tmpelt;
      }
    }
    
    // calculate new attr
    console.log("------------------------ Getting new axis vector ------------------------------")
    var V = {}, Vchanged = {}, Verror = {}, norm = 0;
    for (var i = 0; i<attrNo; i++) {
      V[attr[i]] = 0;
      Vchanged[attr[i]] = 0;
    }

    for (var i = 0; i<attrNo; i++) {
     V[attr[i]] = x0[attr[i]]-x1[attr[i]];
     norm = norm + (x0[attr[i]]-x1[attr[i]])*(x0[attr[i]]-x1[attr[i]]);
    }
    var VV = [];
    for (var i = 0; i<attrNo; i++) {
      VV[i] = {"attr":attr[i], "value":V[attr[i]]};
    }
    VV.sort(function(a,b) {return Math.abs(b["value"]) - Math.abs(a["value"]);});
    for (var i = 0; i<VV.length; i++) {

    }
    norm = Math.sqrt(norm);
    for (var i = 0; i<attrNo; i++) {
      V[attr[i]] = V[attr[i]]/norm;
      if (hlpair.length>1) { Verror[attr[i]] = d3.deviation(hlpair, function(d) { return d[attr[i]]; }); }
      else { Verror[attr[i]] = 0; }
    }
    console.log("------------------------ Calculating new attr ------------------------------")
  } else {
    var V = givenV, Vchanged = givenVchanged, Verror = {}, norm = 0;
    for (var i = 0; i<attrNo; i++) {
     norm = norm + (V[attr[i]])*(V[attr[i]]);
    }
    norm = Math.sqrt(norm);
    for (var i = 0; i<attrNo; i++) {
     V[attr[i]] = V[attr[i]]/norm;
     Verror[attr[i]] = 0;
    }
    console.log("------------------------ Calculating new attr ------------------------------")
  }

    index = index + 1;
    var newxname = 'Custom Axis';
    graph.points.forEach(function(d,i) {
      d["coord"][newxname] = 0; 
      for (var j = 0; j<attrNo; j++) {
        d["coord"][newxname] = d["coord"][newxname] + V[attr[j]]*d["coord"][attr[j]];
      }
      if (istxtdata==true) {
        d["coord"][newxname] = d["coord"][newxname]==0 ? 0 : Math.sign(d["coord"][newxname])*Math.pow(Math.abs(d["coord"][newxname]),0.5);
      }
    });

    // update IAL weight vector
    ial.usermodel.setAttributeWeightVector(V, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_calc', 'whichAxis': axistobeupdated, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
    LE.log(JSON.stringify(ial.logging.peek()));
    
    // recompute bias metrics
    //updateBias(true);
    console.log("------------------------ Done ------------------------------")


    d3.select("#pcplot").selectAll("svg").remove();
    dims[dims.length] = newxname;
    drawParaCoords(data,dims);

    d3.select("#SC").remove();
    d3.select("#"+axistobeupdated).remove();
    data.forEach(function(d) {d[axistobeupdated=="X" ? "x" : "y"] = d["coord"][newxname]; });
    graph = new SimpleGraph("scplot", data, {
            "xlabel": axistobeupdated == "X" ? newxname : graph.options.xlabel,
            "ylabel": axistobeupdated == "X" ? graph.options.ylabel : newxname,
            "init": false,
            "dropzone": graph.dropzone
          });
    var VV = [];
    for (var i = 0; i<attrNo; i++) {
      VV[i] = {"attr":attr[i], "value":V[attr[i]], "changed":Vchanged[attr[i]], "error":Verror[attr[i]]};
    }
    if (axistobeupdated == "X") { X = VV; xaxis = new axis("#scplot", VV, axistobeupdated, axisOptions[axistobeupdated]); }
    else { Y = VV; yaxis = new axis("#scplot", VV, axistobeupdated, axisOptions[axistobeupdated]) }
};

SimpleGraph.prototype.keydown = function() {
  isDragging = true; 
        dragStart = new Date();

  var self = this;
  return function() {
    if (!self.selected) return;
    switch (d3.event.keyCode) {
      case 8: // backspace
      case 46: { // delete
        var i = self.points.indexOf(self.selected);
        self.points.splice(i, 1);
        self.selected = self.points.length ? self.points[i > 0 ? i - 1 : 0] : null;
        self.update();
        break;
      }
    }
  }
};

SimpleGraph.prototype.redraw = function() {
  var self = this;
  return function() {
    var tx = function(d) { 
      return "translate(" + self.x(d) + ",0)"; 
    },
    ty = function(d) { 
      return "translate(0," + self.y(d) + ")";
    },
    stroke = function(d) { 
      return d ? "#ccc" : "#666"; 
    },
    fx = self.x.tickFormat(10),
    fy = self.y.tickFormat(10);

    // Regenerate x-ticks…
    var gx = self.vis.selectAll("g.x")
        .data(self.x.ticks(10), String)
        .attr("transform", tx);

    gx.select("text")
        .text(fx);

    var gxe = gx.enter().insert("g", "a")
        .attr("class", "x")
        .attr("transform", tx);

    gxe.append("line")
        .attr("stroke", stroke)
        .attr("y1", 0)
        .attr("y2", self.size.height);

    gxe.append("text")
        .attr("class", "axis")
        .attr("y", self.size.height)
        .attr("dy", "1em")
        .attr("text-anchor", "middle")
        .text(fx)
        .style("cursor", "ew-resize")
        .on("mouseover", function(d) { d3.select(this).style("font-weight", "bold");})
        .on("mouseout",  function(d) { d3.select(this).style("font-weight", "normal");})
        .on("mousedown.drag",  self.xaxis_drag())
        .on("touchstart.drag", self.xaxis_drag());

    gx.exit().remove();

    // Regenerate y-ticks…
    var gy = self.vis.selectAll("g.y")
        .data(self.y.ticks(10), String)
        .attr("transform", ty);

    gy.select("text")
        .text(fy);

    var gye = gy.enter().insert("g", "a")
        .attr("class", "y")
        .attr("transform", ty)
        .attr("background-fill", "#FFEEB6");

    gye.append("line")
        .attr("stroke", stroke)
        .attr("x1", 0)
        .attr("x2", self.size.width);

    gye.append("text")
        .attr("class", "axis")
        .attr("x", -3)
        .attr("dy", ".35em")
        .attr("text-anchor", "end")
        .text(fy)
        .style("cursor", "ns-resize")
        .on("mouseover", function(d) { d3.select(this).style("font-weight", "bold");})
        .on("mouseout",  function(d) { d3.select(this).style("font-weight", "normal");})
        .on("mousedown.drag",  self.yaxis_drag())
        .on("touchstart.drag", self.yaxis_drag());

    gy.exit().remove();
    self.plot.call(d3.behavior.zoom().x(self.x).y(self.y).on("zoom", self.redraw()));
    self.update();    
  }  
}

SimpleGraph.prototype.xaxis_drag = function() {
  var self = this;
  return function(d) {
    document.onselectstart = function() { return false; };
    var p = d3.mouse(self.vis[0][0]);
    self.downx = self.x.invert(p[0]);
  }
};

SimpleGraph.prototype.yaxis_drag = function(d) {
  var self = this;
  return function(d) {
    document.onselectstart = function() { return false; };
    var p = d3.mouse(self.vis[0][0]);
    self.downy = self.y.invert(p[1]);
  }
};

function tabulate(dataitem, option) {
    var op = option || "no";
    if (op == "click") { return; var tid = "#datapanel2";}
    else {var tid = "#datapanel";}

    d3.select(tid).selectAll("table").remove();
    
    var columns, thead, tbody, data = [];
    
    if (op == 'empty') // construct an empty table
    	columns = ["Player",""];
    else 
    	columns = [dataitem["Name"],""];
    
    var table = d3.select(tid).append("table")
                .attr("style", "margin-left: 5px"),

	   thead = table.append("thead"),
	   tbody = table.append("tbody");

	// append the header row
	 thead.append("tr")
		.selectAll("th")
		.data(columns)
		.enter()
		.append("th")
	  .text(function(column) { return column;});


	columns = ["key", "value"];
	if (op == 'empty') {
        if (istxtdata==false) {
          for (var i = 0; i<attrNo; i++) {
            var item = {"key":null, "value":null};
            item["key"] = attr[i];
            item["value"] = "";
            data[i] = item;
          }
        } else {
          table.style("width", "810px");
          data[0] = {"key":'{'+dataitem["text"]["title"]+emptyspace+'}', "value":""};
          data[1] = {"key":dataitem["text"]["content"], "value":""};
        }
	} else {
		if (istxtdata==false) {
			for (var i = 0; i<attrNo; i++) {
		        var item = {"key":null, "value":null};
		        item["key"] = attr[i];
		        item["value"] = dataitem[attr[i]];
		        data[i] = item;
		     }
		 } else {
		      table.style("width", "810px");
		      data[0] = {"key":'{'+dataitem["text"]["title"]+emptyspace+'}', "value":null};
		      data[1] = {"key":dataitem["text"]["content"], "value":null};
		 }
	}

    // create a row for each object in the data
    var rows = tbody.selectAll("tr")
        .data(data)
        .enter()
        .append("tr");

 
    // create a cell in each row for each column
    var cells = rows.selectAll("td.data")
        .data(function(row) {
            return columns.map(function(column) {
            return {column: column, value: row[column]};
            });
        })
        .enter()
        .append("td")
        .html(function(d) { return d.value;});

    return table;

}

clearDropzone = function(axistobeupdated) {
  data = graph.points;
  graph.dropzone[axistobeupdated+"L"] = [];
  graph.dropzone[axistobeupdated+"H"] = [];
  var inotherdropzone = graph.dropzone.XH.concat(graph.dropzone.XL,graph.dropzone.YH,graph.dropzone.YL);
  data.forEach(function(d) { d.indropzone = false; inotherdropzone.forEach(function(c) { if (c==d) { d.indropzone = true; return; } }); });
  d3.select("#"+axistobeupdated+"L").selectAll("circle").remove();
  d3.select("#"+axistobeupdated+"H").selectAll("circle").remove();
  
  if (istxtdata==true) {
    if (axistobeupdated == "X") { data.forEach(function(d) {d.x = d["coord"]["tsne_1"]; }); }
    else { data.forEach(function(d) {d.y = d["coord"]["tsne_2"]; }); }
    d3.select("#SC").remove();
    d3.select("#"+axistobeupdated).remove();
    graph = new SimpleGraph("scplot", data, {
            "xlabel": axistobeupdated == "X" ? "tsne_1" : graph.options.xlabel,
            "ylabel": axistobeupdated == "X" ? graph.options.ylabel : "tsne_2",
            "init": axistobeupdated,
            "dropzone": graph.dropzone
          });
    
    // document.getElementById("cb"+axistobeupdated).selectedIndex = attrNo+index-1;

    var V = [];
    for (var i = 0; i<attrNo; i++) {
      V[i] = {"attr":attr[i], "value":0, "changed":0, "error":0};
    }
    if (axistobeupdated == "X") { X = V; xaxis = new axis("#scplot", V, axistobeupdated, axisOptions[axistobeupdated]); }
    else { Y = V; yaxis = new axis("#scplot", V, axistobeupdated, axisOptions[axistobeupdated]); }
  }
  else { 
    document.getElementById("cb"+axistobeupdated).selectedIndex = axistobeupdated=="X" ? initdim1 : initdim2;
    updatebycb(axistobeupdated, axistobeupdated=="X" ? attr[initdim1] : attr[initdim2]); 
  }
  console.log(graph.dropzone);
}

updatebycb = function(axistobeupdated, selectedattr) {
    data = graph.points;
    var V = [], newxname = selectedattr, V2 = {};
    for (var i = 0; i<attrNo; i++) {
      V[i] = {"attr": attr[i], "value": attr[i]==selectedattr ? 1 : 0, "error":0};
      V2[attr[i]] = attr[i]==selectedattr ? 1 : 0;
    }
    for (var i = 0; i<attr2.length; i++) {
      if (attr2[i]["attr"]==selectedattr) {
    	  V = attr2[i]["vector"];
    	  for (var j = 0; j < attr2[i]["vector"].length; j++)
    		  V2[attr2[i]["vector"][j]["attr"]] = attr2[i]["vector"][j]["value"];
      }
    }
    
    // update IAL weight vector
    ial.usermodel.setAttributeWeightVector(V2, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_select', 'whichAxis': axistobeupdated, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
    LE.log(JSON.stringify(ial.logging.peek()));
    
    d3.select("#SC").remove();
    d3.select("#"+axistobeupdated).remove();
    data.forEach(function(d) {d[axistobeupdated=="X" ? "x" : "y"] = d["coord"][newxname]; });
    graph = new SimpleGraph("scplot", data, {
            "xlabel": axistobeupdated == "X" ? newxname : graph.options.xlabel,
            "ylabel": axistobeupdated == "X" ? graph.options.ylabel : newxname,
            "init": axistobeupdated,
            "dropzone": graph.dropzone
          });
    if (axistobeupdated == "X") { X = V; xaxis = new axis("#scplot", V, axistobeupdated, axisOptions[axistobeupdated]);}
    else { Y = V; yaxis = new axis("#scplot", V, axistobeupdated, axisOptions[axistobeupdated]);} 
    
    // clear out the drop zones if axis is being transformed
    data = graph.points;
    graph.dropzone[axistobeupdated+"L"] = [];
    graph.dropzone[axistobeupdated+"H"] = [];
    var inotherdropzone = graph.dropzone.XH.concat(graph.dropzone.XL,graph.dropzone.YH,graph.dropzone.YL);
    data.forEach(function(d) { d.indropzone = false; inotherdropzone.forEach(function(c) { if (c==d) { d.indropzone = true; return; } }); });
    d3.select("#"+axistobeupdated+"L").selectAll("circle").remove();
    d3.select("#"+axistobeupdated+"H").selectAll("circle").remove();
    d3.selectAll(".dropped").classed("dropped", function(d) {
    	if (d.indropzone) return true;
    	else return false;
    });
}

saveAxis = function(axistobeupdated) {
  var newxname = axistobeupdated=="X" ? graph.options.xlabel : graph.options.ylabel;
  var count = 0;
  for (var i = 0; i<attr2.length; i++) {
    if (attr2[i]["attr"]==newxname) {
      count = count + 1;
      break;
    }
  }
  for (var i = 0; i<attr.length; i++) {
    if (attr[i]==newxname) {
      count = count + 1;
      break;
    }
  }
  if (count==0) {
    addNewAxis(newxname, axistobeupdated=="X" ? X : Y);
    setTimeout(function(){document.getElementById("cb"+axistobeupdated).selectedIndex = attrNo+attr2.length-1;}, 3);
  }
}

addNewAxis = function(newxname, newaxisvector) {
  attr2[attr2.length] = {"attr": newxname, "vector": newaxisvector};
  d3.select("#cbY").append("option").attr("value",newxname).text(newxname);
  d3.select("#cbX").append("option").attr("value",newxname).text(newxname);
}

// Updated bias mitigation

// recomputeAttrWeightMetrics: if true, recompute attribute weight metrics
// otherwise, don't update them
/*function updateBias(recomputeAttrWeightMetrics) {
  if (typeof recomputeAttrWeightMetrics == 'undefined')
	  recomputeAttrWeightMetrics = true; 
  console.log("****Updating bias metrics****");
  $("#datapanel2").html("");
  
  var tip = d3.tip()
	  .attr('class', 'd3-tip')
	  .offset([-10, 0])
	  .html(function(d) {
		  return "<strong>" + ((100*d["metric_level"]).toFixed(0)) + "%</strong>";
	  });
  
  // mouseover bias bar
  function mouseover(d) {
	  if (d["bias_type"] == "bias_data_point_coverage") {
		  //console.log("data point coverage");
		  var visitedIds = d["info"]["visited"];
		  
		  // color circles that have been visited
		  d3.selectAll("circle")
		  	.attr("r", function(b) {
		  		if (visitedIds.has(b["ial"]["id"])) return 11.0;
		  		else return 3.0;
		  	});
	  } else if (d["bias_type"] == "bias_data_point_distribution") {
		  //console.log("data point distribution");
		  var maxInt = d["info"]["max_observed_interactions"]; 
		  
		  d3.selectAll("circle")
		  	.attr("r", function(b) {
		  		var observedInt = 0; 
		  		if (d["info"]["distribution_vector"].hasOwnProperty(b["ial"]["id"]))
		  			observedInt = d["info"]["distribution_vector"][b["ial"]["id"]]["observed"];
		  		if (observedInt == 0)
		  			return 3.0;
		  		else {
		  			scaledVal = observedInt / maxInt;
		  			if (scaledVal < 0.25) return 5.0;
		  			else if (scaledVal < 0.5) return 7.0;
		  			else if (scaledVal < 0.75) return 9.0;
		  			else return 11.0
		  		}
		  	});
	  } else if (d["bias_type"] == "bias_attribute_coverage") {
		  //console.log("attribute coverage");
		  
		  d3.select("#X").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal);
		  	});
		  
		  d3.select("#Y").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal); 
		  	});
	  } else if (d["bias_type"] == "bias_attribute_distribution") {
		  //console.log("attribute distribution");
		  
		  d3.select("#X").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal);
		  	});
		  
		  d3.select("#Y").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal); 
		  	});
	  } else if (d["bias_type"] == "bias_attribute_weight_coverage") {
		  //console.log("attribute weight coverage");
		  
		  d3.select("#X").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal);
		  	});
		  
		  d3.select("#Y").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal); 
		  	});
	  } else if (d["bias_type"] == "bias_attribute_weight_distribution") {
		  //console.log("attribute weight distribution");
		  
		  d3.select("#X").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal);
		  	});
		  
		  d3.select("#Y").selectAll(".bar")
		  	.attr("fill-opacity", function(b) {
		  		var metVal = 0; 
		  		if (d["info"]["attribute_vector"].hasOwnProperty(b["attr"]))
		  			metVal = d["info"]["attribute_vector"][b["attr"]]["metric_level"];
		  		return biasOpacityScale(metVal); 
		  	});
	  } 
	  tip.show(d);
  }
  
  // mouseout bias bar
  function mouseout(d) {
	  d3.selectAll("circle")
	  	.attr("r", 7.0)
	  	.classed("visited", false)
	  	.classed("unvisited", false)
	  	.classed("heat0", false)
	  	.classed("heat1", false)
	  	.classed("heat2", false)
	  	.classed("heat3", false)
	  	.classed("heat4", false);
	  d3.select("#X").selectAll(".bar")
	  	.attr("fill-opacity", 1.0);
	  d3.select("#Y").selectAll(".bar")
	  	.attr("fill-opacity", 1.0);
	  tip.hide(d);
  }
  
  var biasSvg = d3.select("#datapanel2").append("svg")
	  .attr("id", "biasVis")
	  .attr("top", $("#datapanel").height())
	  .attr("left", $("#scplot").width())
	  .attr("width", 2*$("#datapanel").width())
	  .attr("height", $("#scplot").height() - $("#datapanel").height());
  
  biasSvg.call(tip);

  var dataPointCoverageResult = ial.usermodel.bias.computeDataPointCoverage(); 
  console.log("Data Point Coverage Result");
  console.log(dataPointCoverageResult);
  //$("#datapanel2").append("<b>Data Point Coverage Metric:</b> " + dataPointCoverageResult['metric_level'].toFixed(4));

  var dataPointDistributionResult = ial.usermodel.bias.computeDataPointDistribution(); 
  console.log("Data Point Distribution Result");
  console.log(dataPointDistributionResult);
  //$("#datapanel2").append("<br><b>Data Point Distribution Metric:</b> " + dataPointDistributionResult['metric_level'].toFixed(4));

  var attributeCoverageResult = ial.usermodel.bias.computeAttributeCoverage(); 
  console.log("Attribute Coverage Result");
  console.log(attributeCoverageResult);
  //$("#datapanel2").append("<br><b>Attribute Coverage Metric:</b> " + attributeCoverageResult['metric_level'].toFixed(4));

  var attributeDistributionResult = ial.usermodel.bias.computeAttributeDistribution(); 
  console.log("Attribute Distribution Result");
  console.log(attributeDistributionResult);
  //$("#datapanel2").append("<br><b>Attribute Distribution Metric:</b> " + attributeDistributionResult['metric_level'].toFixed(4));
  
  var attributeWeightCoverageResult = biasResults[4];
  var attributeWeightDistributionResult = biasResults[5];
  if (recomputeAttrWeightMetrics) {
	  var attributeWeightCoverageResult = ial.usermodel.bias.computeAttributeWeightCoverage();
	  //$("#datapanel2").append("<br><b>Attribute Weight Coverage Metric:</b> " + attributeWeightDistributionResult['metric_level'].toFixed(4));
	
	  var attributeWeightDistributionResult = ial.usermodel.bias.computeAttributeWeightDistribution();
	  //$("#datapanel2").append("<br><b>Attribute Weight Distribution Metric:</b> " + attributeWeightDistributionResult['metric_level'].toFixed(4));
  } 
  
  console.log("Attribute Weight Coverage Result");
  console.log(attributeWeightCoverageResult);
  
  console.log("Attribute Weight Distribution Result");
  console.log(attributeWeightDistributionResult);
  
  biasResults = [dataPointCoverageResult, dataPointDistributionResult, attributeCoverageResult, attributeDistributionResult, attributeWeightCoverageResult, attributeWeightDistributionResult];
  var bars = d3.select("#biasVis").selectAll("g")
	  .data(biasResults)
	  .enter().append("g")
	  .attr("transform", function(d, i) { return "translate(10," + (i * 25 + (i + 1) * 10) + ")"; });
  bars.append("rect")
	  .attr("x", function(d, i) { return 75; })
	  .attr("width", function(d, i) { return biasWidthScale(d["metric_level"]); })
	  .attr("height", 25)
	  .classed("biasBar", true)
	  .classed("bar", true)
	  .classed("bias", true)
	  .on("mouseover", mouseover)
	  .on("mouseout", mouseout); 
  bars.append("text")
	  .attr("x", function(d, i) { return 70; })
	  .attr("y", function(d, i) { return 12; })
	  .attr("dy", ".35em")
	  .text(function(d, i) { if (i == 0) return "Data Cov."; else if (i == 1) return "Data Distr."; else if (i == 2) return "Attr. Cov."; else if (i == 3) return "Attr. Distr."; else if (i == 4) return "Attr. Weight Cov."; else return "Attr. Weight Distr."; })
	  .attr("text-anchor", "end");
  
}*/