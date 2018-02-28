var graph;

var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;

// Gets called when the page is loaded.
function init(){
  // Get input data
  d3.csv('data/04cars data_clean.csv', function(data) {
    var loaddata = jQuery.extend(true, [], data);
    for (var i = 0; i<loaddata.length; i++) {
      loaddata[i]["Name"] = loaddata[i]["Vehicle Name"];
      delete loaddata[i]["Vehicle Name"];
      delete loaddata[i]["Pickup"];
      loaddata[i]["coord"] = {};
    }
    attr = Object.keys(loaddata[0]);
    attr.pop(); // remove "coord" from attribute list
    attr.pop(); // remove "Name" from attribute list
    attrNo = attr.length;
    for (var i = 0; i<attrNo; i++) {
      if (attr[i] != "Name" && attr[i] != "coord") {
        var tmpmax = d3.max(loaddata, function(d) { return +d[attr[i]]; });
        var tmpmin = d3.min(loaddata, function(d) { return +d[attr[i]]; });
        loaddata.forEach(function(d) {
          d["coord"][attr[i]] = (+d[attr[i]]-tmpmin)/(tmpmax-tmpmin);
        });
      }
    }
    ial.init(loaddata, 0, ["coord", "Name"], "exclude", -1, 1);
    console.log("ial initialized");
    loadVis(loaddata);
  });
}

// Main function
function loadVis(data) {
  drawScatterPlot(data);
  updateBias(true);
  showColorRamp();
  
  for (var i = 0; i<attrNo; i++) {
    dims[i] = attr[i];
  }
  drawParaCoords(data,dims);
  tabulate(data[0]);
}

//show color ramp for bias metrics
function showColorRamp() {
	colors = ["#7b3294", "#c2a5cf", "#f7f7f7", "#a6dba0", "#008837"];
	// TODO: create svgs to show high / low color scheme
}

function drawScatterPlot(data) {
  // heterogeneous data
  initdim1 = 13, initdim2 = 11; // 8, 12 = retail price, city mpg
  data.forEach(function(d) {d.x = d["coord"][attr[initdim1]]; d.y = d["coord"][attr[initdim2]]; });
  graph = new SimpleGraph("scplot", data, {
          "xlabel": attr[initdim1],
          "ylabel": attr[initdim2],
          "init": true
        });
  
  var V1 = {}, V2 = {};
  for (var i = 0; i<attrNo; i++) {
    X[i] = {"attr":attr[i], "value":0, "changed":0, "error":0};
    Y[i] = {"attr":attr[i], "value":0, "changed":0, "error":0};
    V1[attr[i]] = 0;
    V2[attr[i]] = 0;
  }
  
  V1[attr[initdim1]] = 1;
  V2[attr[initdim2]] = 1;
  
  //update IAL weight vector
  ial.setAttributeWeightVector(V1, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init'});
  ial.setAttributeWeightVector(V2, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init'});

  X[initdim1]["value"] = 1;
  Y[initdim2]["value"] = 1;
  document.getElementById("cbX").selectedIndex = initdim1;
  document.getElementById("cbY").selectedIndex = initdim2;

  xaxis = new axis("#scplot", X, "X", {
          "width": graph.size.width-dropSize*2,
          "height": graph.padding.bottom-40,
          "padding": {top: graph.padding.top+graph.size.height+40, right: 0, left: graph.padding.left+dropSize+10, bottom: 0}
        });
  yaxis = new axis("#scplot", Y, "Y", {
          "width": graph.padding.left-dropSize,
          "height": graph.size.height-dropSize*2,
          "padding": {top: graph.padding.top+dropSize, right: 0, left: 15, bottom: 0}
        });
}