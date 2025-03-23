# NetworkAPI Engine

This page is a Work-in-progress.

## Concepts
The Network class is the focus for the NetworkAPI.  It is a container for all of the components that make up your app or experiment and is a framework to orchestrate the execution of that experiment.

Regions are a unit of execution, like a building block.  Each region wrappers one of the basic algorithms and marshals the parameters needed by that algorithm. The fundamental regions are built-in to the htm.core library.  External regions can also be registered and with the Network class to extend the set of building blocks available.

Links define the data flow from one region building block to another.  It provides coordination and data conversion if needed.

There needs to be an app to act as the main for your experiment. This will allocate the Network class, configure it with a list of regions and links. It can then tell the Network class to execute an iteration.  Between iterations the app can feed data into the network and sample data from the network.

To illustrate here is a simple example with 3 regions.
```
    ///////////////////////////////////////////////////////////////
    //
    //                          .------------------.
    //                         |    encoder        |
    //       sine wave data--->|(RDSEEncoderRegion)|
    //                         |                   |
    //                         `-------------------'
    //                                 |
    //                         .-----------------.
    //                         |     sp          |
    //                         |  (SPRegion)     |
    //                         |                 |
    //                         `-----------------'
    //                                 |
    //                         .-----------------.
    //                         |      tm         |
    //                         |   (TMRegion)    |---->anomaly score
    //                         |                 |
    //                         `-----------------'
    //
    //////////////////////////////////////////////////////////////////
```
In our app we first allocate a Network object and configure it with the topology and parameters for an experiment.  Then in a loop we repeatedly execute the experiment, feeding it data during each iteration. Then we capture the resulting data.

```
  import numpy as np
  import math
  from htm.bindings.engine_internal import Network,Region
  
  EPOCHS = 5000
  
  net = Network()
  config = """
    {network: [
      {addRegion: {name: "encoder", type: "RDSEEncoderRegion", params: {size: 1000, sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}}},
      {addRegion: {name: "sp", type: "SPRegion", params: {columnCount: 2048, globalInhibition: true}}},
      {addRegion: {name: "tm", type: "TMRegion", params: {cellsPerColumn: 8, orColumnOutputs: true}}},
      {addLink:   {src: "encoder.encoded", dest: "sp.bottomUpIn"}},
      {addLink:   {src: "sp.bottomUpOut", dest: "tm.bottomUpIn"}}
     ]}"""
  net.configure(config)

  # iterate EPOCHS times
  x = 0.00
  for e in range(EPOCHS):
      s = math.sin(x) 
      net.getRegion('encoder').setParameterReal64('sensedValue', s) # feed data to encoder
      net.run(1)
      x += 0.01  # advance one step, 0.01 radians

  score = np.array(net.getRegion('tm').getOutputArray('anomaly'))
  print(score[0])
```

## Network class
```
C++:  #include <htm.engine.Network.hpp>

Python: from htm.bindings.engine_internal import Network
```
The most commonly used methods:

| Method                                   | Description                              |
| Network( )                               | Constructor                              |
| configure(json_string)                 | Configures the Network object with links and regions with parameters.|
| addRegion(region_name, region_type, parameters) | Configures a region instance of the given region type and parameters. |
| link(source_region, dest_region, &quot;&quot;, &quot;&quot;, source_output, dest_input) | Configures a link between an output on the source region to an input on the destination region. |
| run(iterations) | Executes all regions |
 
 
Less used support methods:

| Method                                   | Description                              |
|------------------------------------------|------------------------------------------|
| registerRegion(region_type,  class wrapper)  for C++<br>registerPyRegion(module, className)  for Python | Register a custom region as a plugin.    |
| unregisterRegion(region_type)            | Unregister a custom region.              |
| getRegistrations()                       | returns JSON list of registered classes  |
| getSpecJSON(region_type)                 | Obtain the Spec for a region type which shows expected parameters, inputs, outputs, and commands. |
| initialize()                             | Signals end of configuration.            |
| setInputData(source_name, data)          | Allows the app to inject array data into a link.<br>(see Region class for get methods.) |
| getRegions( )                            | Retrieve a list of active region objects on the Network object. |
| getRegion(region_name)                   | Retrieve a specific region instance by name. |
| removeRegion(region_name)                | Remove the region instance from the Network object. |
| getLinks( )                              | Retrieve a list of active links on the Network object. |
| removeLink(source_region, dest_region, sourceOutput, destInput) | Remove the link from the Network object. |
| getExecutionMap()                        | Returns a string showing the order of execution and links that propogate data.  Useful for debugging. |
| setPhases(region_name, phase_number)     | Override the run execution order for a region. |
| getPhases(region_name)                   | Retrieve the current phases specified for a region. |
| save() <br/> load() <br/> pickle (for Python only)                          | Serialize and deseerilaize a Network to/from a stream. |
| saveToFile(filename, format) loadFromFile(filename, format)            | Serialize and deserialize a Network to/from a file. |
| setLogLevel(level)                       | Turn on/off region execution logging.    |


### Network()
The primary constructor has no arguments. Internally there are constructors for deserialization and moving but it cannot be copied.

### Network::configure(json_string)
Configure the Network object with a single text string.
```
  @param json_string
    A YAML or JSON string that defines regions and links.
    Additional regions and links can be added with the addRegion() and Link() methods.
     
    YAML Syntax:
         network:
            - addRegion:
                name: <region_name>
                type: <region_type>
                params: <parameters>  
                phase:  <phase_number>
   
            - addLink:
                src: <region_name> "." <output_name>
                dest: <region_name"." <input_name>
                delay: <iterations to delay>
   
    JSON syntax:
    {network: [
      {addRegion: {name: <region_name>, type: <region type>, params: {<parameters>}, phase: <phase>}},
      {addLink:   {src: "<region_name>.<output_name>", 
                   dest: "<region_name>.<input_name>", delay: <delay>}},
    ]}
```
'**region_type**' identifies the region implementation. This can be a [NetworkAPI Built-in Region](NetworkAPI_Regions.md) or the name of a [registered custom region](NetworkAPI_Creating_Regions.md). Custom regions can be implemented in C++ or Python. The type name of a custom region implemented in Python is the class name with "py." prepended.

'**region_name**' is an identifier you choose for this specific instance of the region.

'**parameters**' is a map of parameters to be passed to the region's underlining algorithm. Each region maintains a list of parameters it expects. See Network::getSpecJSON(region_type) or Region::getSpec().
Default values will be provided for any parameters not specified in the 'parameters'.  Optional, default is empty string.

'**phase**' identifies the order in which regions will be executed for an iteration. All regions with the same phase number will be executed in the order they are declared. If the phase number is not given, it will be placed in phase 0. Optional.

'**input_name**' is the name of one of the region inputs as defined in the Spec for that region.

'**output_name**' is the name of one of the region outputs as defined in the Spec for that region.

'**delay**' is the number of iterations to delay the flow of data over that link.  Initial values will be all 0s.  Optional, default=0.

See [NetworkAPI Built-in Regions](NetworkAPI_Regions.md) for more details on the built-in links available and [NetworkAPI How to create Regions](NetworkAPI_Creating_Regions.md) for creating custom regions.

See [NetworkAPI Links](NetworkAPI_Links.md) for more details on how to configure and use links.

### Network::addRegion(region_name, region_type, parameters, phases)
An alternative way to declare a region.

@param **region_name**, a string, a name you are assigning to this instance of the region.

@param **region_type**, a string, the name of the region implementation. This can be a [NetworkAPI Built-in Region](NetworkAPI_Regions.md) or the name of a [registered custom region](NetworkAPI_Creating_Regions.md). Custom regions can be implemented in C++ or Python.

@param **parameters**, a JSON string containing parameters to pass to the underlining algorithm. The region Spec will declare all parameters that can be used. Default values will be provided for any parameters not specified in the 'parameters'.  For C++ users this can be a pre-parsed ValueMap object.

@param **phases**, an ordered set of phaseID's for the phases that this region will be placed. Optional, by default the region will be placed into phase 0.

See [NetworkAPI Built-in Regions](NetworkAPI_Regions.md) for more details on the built-in links available and [NetworkAPI How to create Regions](NetworkAPI_Creating_Regions.md) for creating custom regions.

### Network::link(srcName, destName, linkType, linkParams, srcOutput, destInput, delay)
An alternative way to declare a link.

@param **srcName**, a string, name of the region instance that is the source of data.  

The special case name "INPUT" indicates that the source will be coming from the app by calling Network::setInputData(sourcename) rather than from a region. For this special case, 'sourcename' is the name given as 'srcOutput' in this link() call.

@param **destName**, a string, the name of the region instance that is the destination of data.

@param **linkType**, empty string, not used. For backward compatibility.

@param **linkParams**, string, parameters are 'dim' and 'mode'.  The 'mode' parameter value can be 'overwrite' which means multiple links to an input will all start writing at bit 0. The default 'mode' value is 'fanIn' which means the buffers from each link be written to a portion of the input buffer so that they do not overlap.  If only one link goes to the input then 'mode' is ignored.
If used with the special srcName of "INPUT" a dim parameter is required to specify the dimensions for the data on the link.  The parameter 'dim' is a JSON string `{"dim": [n, m, j]}` where n, m and j are one, two, or three dimensions indicating the size and shape of the link data. 

@param **srcOutput**, string, identifies the output on the source region where data will be obtained.

@param **destInput**, string, identifies the input of the destination region where data will be sent.

See [NetworkAPI Links](NetworkAPI_Links.md) for more details on how to configure and use links.
            
### Network::run(iteration, phases)
This will execute all regions in the order of declaration unless overridden by the phase attribute.  Just after the region is executed, the output buffers of any connected links is moved to the input buffers of the target region.  Normally a shared pointer is passed from the output to the input so the data is not physically copied.  The data is copied in these links only if a propagation delay is configured, the link is a Fan-In (multiple outputs to a single input and not 'overwrite' mode) or a data type conversion is required.

@param **iteration**, number, the number of iterations to execute before returning.

@param **phases**, a list of phases to execute in this order.  Optional.  If not given all phases are executed.

`#####################################################################`

### Network::registerRegion(region_type, region_wrapper)
For C++ only.  Registers a custom region type for a region written in C++ and linked with the app. A custom building block to expand the pallet of region types beyond those that are built-in.  `Network::registerRegion("MyRegion", new RegisteredRegionImplCpp<MyRegion>());`

@param region_type, string, the name to be assigned to the region type (normally the class name).  

@param region_wrapper, a RegisteredRegionImplCpp object that knows how to instantiate the object and implement the plugin protocol.

### Network::registerPyRegion(module, class_name)
For Python only.  Registers a custom region type for a region written in Python. A custom building block to expand the pallet of region types beyond those that are built-in.

@param module, string, the path to the module that is to be loaded.

@param class_name, string, the name of the class that is to be registered.
The registered region_type for this region will be the class name with "py." prepended.

### Network::unregisterRegion(region_type)
Erase the registration for a custom region type. This works for both C++ registered custom regions and those registered via `registerPyRegion(module, class_name)`.

@param region_type, string, the name of the region type as defined by the registration.

### Network::getRegistrations()
@returns A JSON string containing a list of registered region types. This includes the built-in regions as well as the registered custom regions written in both C++ and Python.

### Network::getSpecJSON(region_type)
Retrieve a JSON string containing the Spec for the specified region type.  The Spec is a static structure that describes what the region expects in terms of parameters, inputs, outputs, and commands.

@param region_type, string, the name of the region type.  See (Built-in Regions)[NetworkAPI_Regions.md].

@returns a JSON string containing a JSON encoding of the Spec.  The fields in the Spec structure are explained in the document (NetworkAPI_Creating_Regions)[NetworkAPI_Creating_Regions.md].

### Network::initialize()
This signals the end of the configuration.  This will be automatically called by run(). After this point the configuration cannot be changed.  The engine will resolve all link end-points, resolve dimensions for inputs and outputs and allocate all buffers.

### Network::setInputData(source_name, data)
Allows an app to inject array data directly into a region's input stream as if it were a link.   See (NetworkAPI Links)[NetworkAPI_Links.md] for more details on how this is used with the special link with "INPUT" as the source region name.

@param source_name, string, name of source paired with "INPUT" in a link. There can be multiple links defined with "INPUT" as the source region as long as the source_name is unique.

@param data, for C++ this is an Array object or a parsed JSON Value object. For Python this is a numpy array.  This is the array to be used as the source data.



### Network::getRegion(region_name)
Fetch the Region object for this region instance.

@param **region_name** a string, the name of the region to obtain.

@returns a Region instance reference. For C++ this is a shared pointer.

### Network::getRegions()
Fetch a list of active regions.

@returns for C++ it is a Collection of Region objects, for Python it is a list of Region objects.

### Network::getRegion(region_name)
Fetch a specific active region object by name.

@param region_name, string, the name of the region to return.

@returns a Region object.

### Network::removeRegion(region_name)
Remove a region instance with that name and any links associated with it.

@param **region_name** a string, the name of the region instance to remove.

### Network::getLinks()
Returns a list of configured links.

@returns this is a vector of shared_ptrs to Link objects.

### Network::removeLink(source_region, dest_region, sourceOutput, destInput)
Remove the link with the specified end points.

@param source_region, dest_region, sourceOutput, destInput; see description of the parameters on the link() method.

### Network::setPhases(region_name, phases)
Override the default sequence of execution.  By default, regions are assigned phase numbers sequentially in the order that they are declared, the first is phase 1, the second is phase 2, etc. The engine executes regions in phase order. As a consequence, by default, regions are executed in the order that they are declared.  By using this method, you can change the order that regions are executed.

@param region_name, string, the name of the region for which the phase is being set.

@param phases, array of ints, The phase into which this region is to be placed.  More than one region can be in the same phase.  A region may also be in more than one phase which means that during a run iteration, that region will be executed once for each phase for which it is a member.

### Network::getPhases(region_name)
Returns the list of phase numbers for the specified region.

@param region_name, string, the name of the region instance for which phases are to be returned.

@returns An array of ints, one element for each phase for which this region is a member.

### Network\::save(out_stream, fmt), <br/>Network::load(in_stream, fmt), <br/>Network\::saveToFile(filename, fmt),<br/>loadFromFile(filename, fmt)

Performs serialization using Cereal.   see: http://uscilab.github.io/cereal/serialization_archives.html
<br/>pickle on Python will indirectly call save() and load() with BINARY fmt.

@param out_stream, an open stream, The stream to send the serialization to. NOTE: for BINARY and PORTABLE the stream must be defined as ios_base::binary or it will crash on Windows. 

@param in_stream, an open stream, The stream to deserialize from.

@param filename, string, the path to the archive being saved or loaded.

@param fmt, enum, the identifier for the mode of the serialization. The default is BINARY. The types are:  
 *   BINARY   - A binary format which is the fastest but not portable between platforms (default).
 *   PORTABLE - Another Binary format, not quite as fast but is portable between platforms.
 *   JSON     - Human readable JSON text format. Slow.
 *   XML      - Human readable XML text format. Even slower.

`============================================================`

## Region class
The Region class represents one of the htm building blocks. 
The class is a generic facade class that presents a uniform interface to apps but the real work is performed by Region implementations that subclass RegionImpl. This allows custom regions to be constructed and registered as plugins. Once registered they behave the same as the built-in implementations. The htm.core library contains a set of built-in Region implementations that can be found in [NetworkAPI_Regions.md](NetworkAPI_Regions.md).

To retain the plug-in paradigm apps must not call the region implementations directly.  All calls to implementations should be made via the common interface of the Region class and all Region classes must be created via Network:addRegion(region_name, region_type, parameters) or the 'addRegion' entry in the config string passed to Network::configure(config).

Member functions used by an app:

 Method  | Description
-------------------------|------------------------------------------------------- 
 getNetwork( ) | returns a reference to the parent Network class 
 getName( )    | returns the name of the region instance 
 getType( )    | returns the name of the Region type (or implementation class) 
 getSpec( )    | returns the raw Spec object for this implementation. 
  isParameter(name) | returns True if parameter exists.
 ***Getters***: <br/> getParameterBool(name)<br/> getParameterByte(name)<br/> getParameterInt32(name)<br/> getParameterUInt32(name)<br/> getParameterInt64(name)<br/> getParameterUInt64(name)<br/> getParameterReal32(name)<br/> getParameterReal64(name)<br/> getParameterString(name) for C++<br/><br/> getParameter(name) for Python <br/><br/> getParameters() <br/> getParameterJSON(name)  for both C++ and Python.<br/><br/> getParameterArray(name, array)<br/> getParameterArrayCount(name) for Arrays.	| returns the value of the specified parameter from this region. 
 ***Setters***:<br/> setParameterBool(name, value)<br/> setParameterByte(name, value)<br/> setParameterInt32(name, value)<br/> setParameterUInt32(name, value)<br/> setParameterInt64(name, value)<br/> setParameterUInt64(name, value)<br/> setParameterReal32(name, value)<br/> setParameterReal64(name, value)<br/> setParameterString(name, value)<br/>  setParameterJSON(name, value)<br/> setParameterArray(name, array) | Sets the value on the specified parameter. The parameter must be declared with ReadWrite access mode in the Spec and the type of the parameter must match the type on the function name. setParameterJSON(name, value) can come from any type. 
getInputData(input_name) | Returns an input buffer as an Array object
getOutputData(output_name) | Returns an output buffer as an Array object
executeCommand(args) | Executes a command on the region.




### Region\::getNetwork()
Backward reference to the Network that created the region.

@returns a reference (a pointer in C++) to the parent Network object. Do not delete or free this pointer.

### Region\::getName()
The Region's instance name. This is the name that is the first argument of the addRegion() call.

@returns, string, The identifier for this instance of the region.  

### Region\::getType()
The type\_name of the Region's implementation (usually its class name).  This is the second argument of the addRegion() call.  This can be a [NetworkAPI Built-in Region](NetworkAPI_Regions.md) or the name of a [registered custom region](NetworkAPI_Creating_Regions.md). Custom regions can be implemented in C++ or Python. The type name of a custom region implemented in Python is the class name with "py." prepended.

@returns, string, the identifier for the instance's implementation.

### Region\::getSpec()
Gets the Spec object for a region's implementation. This is a static structure created by every region implementation describing the parameters, inputs, outputs and commands that it supports. This is read from the regions and cached in the engine. Also see Network::getSpecJSON(region_type) for a JSON string version of this structure.

@returns a reference (std::shared_ptr<Spec> for C++) for the static 

### Region\::getParameterXXX(name) C++ <br/> Region\::getParameter(name) Python

### Region\::getParameters()

### Region\::getParameterJSON(name)

### Region\::getParameterArray(name, array)

### Region\::getParameterArrayCount(name)

### Region\::setParameterXXX(name, value) where XXX is the type

### Region\::setParameterJSON(name, value)

### Region\::setParameterArray(name, array)

### Region\::getInputData(input_name)

### Region\::getOutputData(output_name)

### Region\::executeCommand(args)


`============================================================`

## Array class




