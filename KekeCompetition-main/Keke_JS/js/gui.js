//var socket = io();

//import simulation code
var simjs = require('./simulation')


//map attributes
var game_state = {};


var action_trans = {
	'r':'right',
	'l':'left',
	'u':'up',
	'd':'down',
	's':'space'
}



/////////////////////////        MAP FUNCTIONS       ///////////////////////

// INITIALIZE THE MAP LOCALLY
function initMap(ascii_map){
	//setup state
	simjs.setupLevel(simjs.parseMap(ascii_map));
	game_state = simjs.getGamestate();
}

// UPDATE THE ASCII MAP BY FEEDING THE SOLUTION THROUGH
function updateMap(step){
    console.log("Updating map with step: ", step);  // Log the step received
    let res = simjs.nextMove(action_trans[step], game_state);

    // Update state and map
    game_state = res['next_state'];
    ascii_map = simjs.doubleMap2Str(game_state['obj_map'], game_state['back_map']);
    console.log("Updated map: ", ascii_map);  // Log the updated ASCII map

    // Check for terminal win state
    return { 'ascii_map': ascii_map, "won": res['won'] };
}

module.exports = {
    updateMap: function(step) { return updateMap(step); }
};


//////////////////////////      MAIN FUNCTIONS      ////////////////////////////

// INITIALIZATION FUNCTION
// function init(am,sol){
// 	//import the images, map, and solution
// 	makeImgHash();
// 	orig_map = am;
// 	keke_solution = sol;
// }



module.exports = {
	initMap : function(m){initMap(m);},
	updateMap : function(step){return updateMap(step);}
}
