var width=100;
var height=200;

var svg= d3.select("#bias")
			.append("svg")
			.attr("width",width)				
			.attr("height",height)
			.attr("padding",5)


var rand = d3.random.normal (50,50)

var dataset=[]

for(var i=0; i<100;i++) {
	dataset.push(rand());
}

console.log(dataset);

var binNum=10,
	rangeMin=0,
	rangeMax=100;

var histogram=d3.layout.histogram()
				.range([rangeMin,rangeMax])
				.bins(binNum)
				.frequency(true);

var hisData=histogram(dataset);

console.log(hisData);

var xAxisWidth=100;
	xTicks=hisData.map(function(d){return d.x});

var xScale=d3.scale.ordinal()
					.domain(xTicks)
					.rangeRoundBands([0,xAxisWidth],0.1);
var yAxisWidth=100;

var yScale= d3.scale.linear()
			.domain([ d3.min(hisData, function(d){ return d.y; }),
					  d3.max(hisData, function(d){ return d.y; }) ])
				.range([5,yAxisWidth]);

var padding = { top: 30 , right: 30, bottom: 30, left: 30 };

var xAxis = d3.svg.axis()
				.scale(xScale)
				.orient("bottom")
				.tickFormat(d3.format(".0f"));

svg.append("g")
	.attr("class","axis")
	.attr("transform","translate("+padding.left+","+(height-padding.bottom)+")")
	.call(xAxis);


console.log(xScale.range());
console.log(xScale.rangeBand());
console.log(xScale(70));
console.log(xScale(94));

var lineGenerator = d3.svg.line()
					.x(function(d){ return xScale(d.x); })
					.y(function(d){ return height - yScale(d.y); })
					.interpolate("basis");

var gLine = svg.append("g")
			.attr("transform","translate(" + padding.left + "," + ( -padding.bottom ) +  ")")
			.style("opacity",1.0);

gLine.append("path")
.attr("class","linePath")
.attr("d",lineGenerator(hisData));
	