var graph;

var defaultX = 0, defaultY = 1;
var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;
var colorScale = d3.scale.category10(); 
var label = {DEFAULT: "Un-Assign", PG: "Point Guard", SG: "Shooting Guard", SF: "Small Forward", PF: "Power Forward", C: "Center"};
var positions = [label.DEFAULT, label.PG, label.SG, label.SF, label.PF, label.C];
var colorMap = {"Un-Assign": "#7f7f7f", "Point Guard": colorScale(0), "Shooting Guard": colorScale(1), "Small Forward": colorScale(2), "Power Forward": colorScale(3), "Center": colorScale(4)};
var defaultColor = colorScale(5);
defaultColor = colorScale(6);
defaultColor = colorScale(7);
var positionDescriptions1 = {"Un-Assign": "un-assign a point", "Point Guard": "usually the smallest and quickest players", "Shooting Guard": "typically of small-medium size and stature", "Small Forward": "typically of medium size and stature", "Power Forward": "typically of medium-large size and stature", "Center": "typically the largest players on the team"};
var positionDescriptions2 = {"Un-Assign": "un-assign a point", "Point Guard": "skilled at passing and dribbling; primarily responsible for distributing the ball to other players resulting in many assists", "Shooting Guard": "typically attempts many shots, especially long-ranged shots", "Small Forward": "typically a strong defender with lots of steals", "Power Forward": "typically spends most time near the basket, resulting in lots of rebounds", "Center": "responsible for protecting the basket, resulting in lots of blocks"};
//var positionDescriptionsDemo = {"Un-Assign": "un-assign a point", "Point Guard": "responsible for controlling the ball", "Shooting Guard": "guards the opponent's best perimeter player on defense", "Small Forward": "typically makes many rebounds", "Power Forward": "typically some of the physically strongest players on the team", "Center": "typically relied on for both strong offense and defense"};
var positionDescriptions;// = positionDescriptions1;
var activePosition = "none";
var playerPositionMap = {};
var helpMouseoverStart = 0; 
var helpMouseoverEnd = 0; 
var ignoreddata=[];
var focuseddata=[];
var u=0;

//Gets called when the page is loaded.
function init(){
	defaultX = $("#initX :selected").text();
	defaultY = $("#initY :selected").text();
	$("#area1").show();
	$("#area2").show();
	$("#area3").show();
	$("#dialog").hide();
	loadData();
}
function loadData() {
	// Get input data
	d3.csv('data/bball_top100_decimal.csv', function(data) { // demo50 dataset
		var loaddata = jQuery.extend(true, [], data);
		for (var i = 0; i < loaddata.length; i++) {
			loaddata[i]["Name"] = "Player " + loaddata[i]["Player Anonymized"];
			delete loaddata[i]["Player"];
			delete loaddata[i]["Player Anonymized"];
			delete loaddata[i]["Position"];
			delete loaddata[i]["Team"];
			playerPositionMap[loaddata[i]["Name"]] = "none";
			loaddata[i]["coord"] = {};
		}

		attr = Object.keys(loaddata[0]);
		attr.pop(); // remove "coord" from attribute list
		attr.pop(); // remove "Name" from attribute list
		attrNo = attr.length;

		// set default user labels
		loaddata.forEach(function(d) {
			d["coord"]["userlabel"] = label.DEFAULT;
		});


		for (var i = 0; i<attrNo; i++) {
			if (attr[i] != "Name" && attr[i] != "coord") {
				var tmpmax = d3.max(loaddata, function(d) { return +d[attr[i]]; });
				var tmpmin = d3.min(loaddata, function(d) { return +d[attr[i]]; });

				// jitter the value by adding a random number up to +/- 3% of scale
				function jitter(val, attribute) {
					var mult = 0.03;
					if (attribute == "Height (Inches)") mult = 0.02;
					if (attribute == "Weight (Pounds)") mult = 0.01;
					var noise = Math.random() * mult * (tmpmax - tmpmin);

					// determine whether to add or subtract
					var sign = Math.round(Math.random());
					if (sign == 0) return val - noise;
					else return val + noise;
				}

				loaddata.forEach(function(d) {
					d["coord"][attr[i]] = jitter(+d[attr[i]], attr[i]);// jitter((+d[attr[i]]-tmpmin)/(tmpmax-tmpmin), attr[i]);
				});
			}
		}

		//determine biased data
		 $(function biasdata(){
      		$('.custom-modal').click(
      			function(){

			 	var aFruits = ["Data Point Coverage Bias is Detected.","Data Point Distribution is Detected","Attribute Coverage Bias is Detected","Attribute Distribution Bias is Detected","Attribute Weight Coverage Bias is Detected","Attribute Weight Distribution Bias is Detected"];
      			u= parseInt(Math.random() * aFruits.length);

      			$('#myModal').find('.bias_type').text(aFruits[u]);

      			if(u<2) {

      				if (u==0){
      				 $('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')
      				ignoreddata=[];
      				for (var m=0; m<5; m++){
			 		var q = parseInt(Math.random() * loaddata.length);
			 		ignoreddata.push(loaddata[q].Name)}
			 		$('#myModal').find('.data').text("You are ignoring the following data points"+ ignoreddata)
					console.log(u)
			 		$('#myModal').find('.focus-data').remove()}

      				else {
		 			    $('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')

					console.log(1)
      				ignoreddata=[];
      				for (var m=0; m<5; m++){
			 		var q = parseInt(Math.random() * loaddata.length);
			 		ignoreddata.push(loaddata[q].Name)
			 		$('#myModal').find('.ignore-data').text("You are ignoring the following data points"+ ignoreddata)
			 		$('#myModal').find('.focus-data').text()}

      				focuseddata=[];
      				for (var n=0;n<5;n++){
			 		var p = parseInt(Math.random() * loaddata.length);
			 		focuseddata.push(loaddata[p].Name)
			 		$('#myModal').find('.focus-data').text("You are focusing too much on the following data points"+ focuseddata)}
      				}

      			} else {
      					if (u==2){
      					$('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')

						console.log(2)
      					var w = parseInt(Math.random() * attr.length);
						var quatile=["lowest","middle","highest"]; 
      					var t = parseInt(Math.random() * quatile.length);
						$('#myModal').find('.ignore-attribute-distribution').text(" You are ignoring "+quatile[t]+" part of distribution of the attribute: "+ attr[w])

						} else if (u==3){
						
						$('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')

						console.log(3)

						var w = parseInt(Math.random() * attr.length);
						var quatile=["lowest","middle","highest"]; 
      					var t = parseInt(Math.random() * quatile.length);
						$('#myModal').find('.ignore-attribute-distribution').text(" You are ignoring "+quatile[t]+" part of the DISTRIBUTION of attribute: "+ attr[w])

      					var d= parseInt(Math.random() * quatile.length);
						$('#myModal').find('.focus-attribute-distribution').text(" You are fosuing too much on "+quatile[d]+" part of the DISTRIBUTION of attribute "+ attr[w])
      					} else 
      					{
						if(u==4){
						$('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')

      					console.log(4)
      					var w = parseInt(Math.random() * attr.length);
						var quatile=["lowest","middle","highest"]; 
      					var f = parseInt(Math.random() * quatile.length);
						$('#myModal').find('.ignore-attribute-weight').text(" You are ignoring "+quatile[f]+" part of the WEIGHT of the attribute:  "+ attr[w])
						} 
						else {

						console.log(5)
						 $('#myModal').find('.focus-data').html('')
		      			$('#myModal').find('.ignore-data').html('')
		      			$('#myModal').find('.ignore-attribute-distribution').html('')
		      			$('#myModal').find('.focus-attribute-distribution').html('')
		      			$('#myModal').find('.ignore-attribute-weight').html('')
		      			$('#myModal').find('.focus-attribute-weight').html('')

						var w = parseInt(Math.random() * attr.length);
						var quatile=["lowest","middle","highest"]; 
      					var f = parseInt(Math.random() * quatile.length);
						$('#myModal').find('.ignore-attribute-weight').text(" You are ignoring "+quatile[f]+" part of the WEIGHT of the attribute "+ attr[w])

						var d= parseInt(Math.random() * quatile.length);
						$('#myModal').find('.focus-attribute-weight').text(" You are fosuing too much on "+quatile[d]+" part of the WEIGHT of the attribute "+ attr[w])}
				}	
						
      						
			 	}
			 })})

		// determine which condition the user follows and which set of descriptions should be used
		if (window.localStorage.getItem("whichCondition") == 1) positionDescriptions = positionDescriptions1; 
		else positionDescriptions = positionDescriptions2;
		//positionDescriptions = positionDescriptionsDemo;

		// initialize log entries
		LE.init('56eed65e-5392-4b18-a6a9-3a10f4b2d753');
		console.log("log entries initialized");

		// initialize IAL
		ial.init(loaddata, 0, ["coord", "Name"], "exclude", -1, 1);
		console.log("ial initialized");

		// load the vis
		loadVis(loaddata);
	});
}

//Main function
function loadVis(data) {
	drawScatterPlot(data);
	//updateBias(true);

	for (var i = 0; i<attrNo; i++) {
		dims[i] = attr[i];
	}
	drawParaCoords(data,dims);
	tabulate(data[0], 'empty');
	addHelp();
	addCustomAxisDropDownControls();
}

function addCustomAxisDropDownControls() {
	$("#cbX").on('change', function() {
	    if ($(this).val() != "Custom Axis") {
	    	$("#cbX option[value='Custom Axis']").remove();
	    }
	});
}

function addHelp() {
	var tooltipText = '<div class="qtip-dark"><b>Task:</b> Your task is to classify all of the points in the scatterplot. Each <i>circle</i> in the scatterplot represents a <i>basketball player</i>. Color each circle according to the <i>position</i> you think the basketball player plays.';
	tooltipText += '<br><br>';
	tooltipText += '<b>Interactions:</b> <ul>';
	tooltipText += '<li><i>See Details</i> about a point by <i>Hovering</i> over it. Details will be shown in the text on the right.</li>';
	tooltipText += '<li><i>Activate a Position</i> by <i>Clicking</i> on a colored circle on the right. A text description of the position will be shown on the bottom right.</li>';
	tooltipText += '<li><i>Deactivate a Position</i> by <i>Double Clicking</i> on any colored circle on the right.</li>';
	tooltipText += '<li><i>Classify a Point</i> on the scatterplot by <i>Clicking</i> on it while its position is activated.</li>';
	tooltipText += '<li><i>Un-Assign a Point</i> on the scatterplot by <i>Clicking</i> on it while "Un-Assign" is activated.</li>';
	tooltipText += '<li><i>Change the Axes</i> by <i>Selecting</i> a new variable from the drop-down on the X or Y axes.</li>';
	tooltipText += '<li><i>Define a Custom Axis</i> by <i>Dragging</i> points from the scatterplot to the bins along the X-Axis.</li>';
	tooltipText += '<li><i>Remove a Point from a Bin</i> on the X-Axis by <i>Double Clicking</i> the point inside the bin.</li>';
	tooltipText += '<li><i>Reset the X-Axis</i> to the default by <i>Clicking</i> the "Clear X" button to clear both bins and change it to the default dimension.</li>';
	tooltipText += '<li><i>Change the Weight of an Attribute</i> along the X-Axis by <i>Dragging</i> the bars to manually change the weight.</li>';
	tooltipText += '</ul>';
	tooltipText += '<br>';
	tooltipText += 'Try to classify all points in the scatterplot to complete the study. When ready to continue, check the box in bottom right and press <span class="studyBlue">Continue</span> to proceed to the next phase of the study.';
	tooltipText += '</div>';
	$("#helpButton").qtip({
		content: {
			title: 'Help:',
			text: tooltipText
		}, 
		style: {
			width: 500,
			classes: 'qtip-dark'
		}
	});
	d3.select("#helpButton").on("mouseover", function() {
			helpMouseoverStart = new Date();
		}).on("mouseout", function() {
			helpMouseoverEnd = new Date();
			// units are seconds -- mouseover time will always encapsulate drag time as well
			var elapsedTime = (helpMouseoverEnd - helpMouseoverStart) / 1000;
			ial.logging.log('help', undefined, 'HelpHover', {'level': 'INFO', 'eventType': 'help_hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
			LE.log(JSON.stringify(ial.logging.peek()));
		});
}


function drawScatterPlot(data) {

	// heterogeneous data
	initdim1 = attr.indexOf(defaultX), initdim2 = attr.indexOf(defaultY); // 1, 2 = height, weight
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
	ial.usermodel.setAttributeWeightVector(V1, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'X', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
	LE.log(JSON.stringify(ial.logging.peek()));
	ial.usermodel.setAttributeWeightVector(V2, true, {'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'Y', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition")});
	LE.log(JSON.stringify(ial.logging.peek()));

	X[initdim1]["value"] = 1;
	Y[initdim2]["value"] = 1;
	document.getElementById("cbX").selectedIndex = initdim1;
	document.getElementById("cbY").selectedIndex = initdim2;

	xaxis = new axis("#scplot", X, "X", {
		"width": graph.size.width-dropSize*2,
		"height": graph.padding.bottom-40,
		"padding": {top: graph.padding.top+graph.size.height+40, right: 0, left: graph.padding.left+dropSize+10, bottom: 0}
	});

//alert
    $(function(){
      $('.custom-modal').click(function biastype(){
      interval = setTimeout(function () {  

		$('#myModal').modal('show');
        
    },1500);   
})
})
}
