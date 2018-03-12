var graph;

var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;
var color = d3.scale.category10();

// Gets called when the page is loaded.
function init(){
  // Get input data
  // var initdim1 = 11, initdim2 = 7;
  // d3.csv('data/04cars data_clean.csv', function(data) {
  //   for (var i = 0; i<data.length; i++) {
  //     var item = {"Name":null, "raw":null, "coord":null};
  //     item["Name"] = data[i]["Vehicle Name"];
  //     item["raw"] = data[i];
  //     delete item["raw"]["Vehicle Name"];
  //     delete item["raw"]["Pickup"];
  //     item["coord"] = {};
  //     loaddata[i] = item;
  //   }
  //   attr = Object.keys(loaddata[0]["raw"]);
  //   attrNo = attr.length;
  //   for (var i = 0; i<attrNo; i++) {
  //     var tmpmax = d3.max(loaddata, function(d) { return +d["raw"][attr[i]]; });
  //     var tmpmin = d3.min(loaddata, function(d) { return +d["raw"][attr[i]]; });
  //     loaddata.forEach(function(d) {
  //       d["coord"][attr[i]] = (+d["raw"][attr[i]]-tmpmin)/(tmpmax-tmpmin);
  //     });
  //   }
  //   loadVis(loaddata);
  // });

  // // Get input data
  // d3.csv('data/crime_clean.csv', function(data) {
  //   for (var i = 0; i<data.length; i++) {
  //     var item = {"Name":null, "raw":null, "coord":null};
  //     item["Name"] = data[i]["communitynamestring"];
  //     item["raw"] = data[i];
  //     delete item["raw"]["communitynamestring"];
  //     delete item["raw"]["state"];
  //     delete item["raw"]["fold"];
  //     item["coord"] = {};
  //     loaddata[i] = item;
  //   }
  //   attr = Object.keys(loaddata[0]["raw"]);
  //   attrNo = attr.length;
  //   for (var i = 0; i<attrNo; i++) {
  //     var tmpmax = d3.max(loaddata, function(d) { return +d["raw"][attr[i]]; });
  //     var tmpmin = d3.min(loaddata, function(d) { return +d["raw"][attr[i]]; });
  //     loaddata.forEach(function(d) {
  //       d["coord"][attr[i]] = (+d["raw"][attr[i]]-tmpmin)/(tmpmax-tmpmin);
  //     });
  //   }
  //   loadVis(loaddata);
  // });

istxtdata = true;
d3.csv('data/infovisvast.csv', function(data) {
    for (var i = 0; i<data.length; i++) {
      var item = {"Name":null, "raw":null, "coord":null, "text":{"title":null, "content":null}, "x":null, "y":null};
      item["Name"] = data[i]["File Name"];
      item["text"]["title"] = data[i]["Paper Title"];
      item["text"]["content"] = data[i]["Paper Content"];
      item["x"] = data[i]["x coord"];
      item["y"] = data[i]["y coord"];
      item["label"] = data[i]["Cluster Membership"];
      item["raw"] = data[i];
      item["coord"] = {"tsne_1":+data[i]["x coord"], "tsne_2":+data[i]["y coord"]};
      delete item["raw"]["File Name"];
      delete item["raw"]["Paper Title"];
      delete item["raw"]["Paper Content"];
      delete item["raw"]["x coord"];
      delete item["raw"]["y coord"];
      delete item["raw"]["Cluster Membership"];
      loaddata[i] = item;
      console.log(loaddata[30])
    }
    attr = Object.keys(loaddata[0]["raw"]);
    attrNo = attr.length;
    loaddata.forEach(function(d) {
      var sum = 0;
      for (var i = 0; i<attrNo; i++) {
        sum = +d["raw"][attr[i]] + sum;
      }
      for (var i = 0; i<attrNo; i++) {
        d["coord"][attr[i]] = d["raw"][attr[i]] / sum;
      }
    });
    loadVis(loaddata);
  });

}


// Main function
function loadVis(data) {
  drawScatterPlot(data);
  for (var i = 0; i<attrNo; i++) {
    dims[i] = attr[i];
  }
  drawParaCoords(data,dims);
  tabulate(data[0]);
}

function drawScatterPlot(data) {
  // heterogeneous data
  // var initdim1 = 11, initdim2 = 7;
  // var initdim1 = 12, initdim2 = 100;
  // data.forEach(function(d) {d.x = d["coord"][attr[initdim1]]; d.y = d["coord"][attr[initdim2]]; });
  // graph = new SimpleGraph("scplot", data, {
  //         "xlabel": attr[initdim1],
  //         "ylabel": attr[initdim2],
  //         "init": true
  //       });
  // text data
  graph = new SimpleGraph("scplot", data, {
          "xlabel": "tsne_1",
          "ylabel": "tsne_2",
          "init": true
        });
  // both
  for (var i = 0; i<attrNo; i++) {
    X[i] = {"attr":attr[i], "value":0, "changed":0, "error":0};
    Y[i] = {"attr":attr[i], "value":0, "changed":0, "error":0};
  }
  // heterogeneous data
  // X[initdim1]["value"] = 1;
  // Y[initdim2]["value"] = 1;
  // document.getElementById("cbX").selectedIndex = initdim1;
  // document.getElementById("cbY").selectedIndex = initdim2;

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