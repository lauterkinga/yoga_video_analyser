äã
¯˙
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ÉĐ

Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:R*
dtype0

Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R*'
shared_nameAdam/dense_23/kernel/v

*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	R*
dtype0

Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_22/kernel/v

*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_21/kernel/v

*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*'
shared_nameAdam/dense_20/kernel/v

*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:"@*
dtype0

Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:R*
dtype0

Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R*'
shared_nameAdam/dense_23/kernel/m

*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	R*
dtype0

Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_22/kernel/m

*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_21/kernel/m

*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*'
shared_nameAdam/dense_20/kernel/m

*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:"@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:R*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	R*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:@*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:"@*
dtype0

serving_default_dense_20_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"*
dtype0*
shape:˙˙˙˙˙˙˙˙˙"
Í
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_20_inputdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1526867

NoOpNoOp
Ų9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*9
value9B9 B9
č
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ļ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Ļ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Ļ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
Ļ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*
* 
°
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 
Ô
;iter

<beta_1

=beta_2
	>decay
?learning_ratemhmimjmk$ml%mm,mn-movpvqvrvs$vt%vu,vv-vw*

@serving_default* 

0
1*

0
1*
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ftrace_0* 

Gtrace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

]0
^1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
_	variables
`	keras_api
	atotal
	bcount*
H
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs*

a0
b1*

_	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

c	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Š
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1527175

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1527284Ā
¤

ø
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
¨

ų
E__inference_dense_22_layer_call_and_return_conditional_losses_1527033

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î

*__inference_dense_22_layer_call_fn_1527022

inputs
unknown:

	unknown_0:	
identityĸStatefulPartitionedCallŪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ë

*__inference_dense_21_layer_call_fn_1527002

inputs
unknown:	@
	unknown_0:	
identityĸStatefulPartitionedCallŪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Į

*__inference_dense_20_layer_call_fn_1526982

inputs
unknown:"@
	unknown_0:@
identityĸStatefulPartitionedCallŨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙": : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
Ņ	
Ã
.__inference_sequential_5_layer_call_fn_1526909

inputs
unknown:"@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	R
	unknown_6:R
identityĸStatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
é	
Ë
.__inference_sequential_5_layer_call_fn_1526790
dense_20_input
unknown:"@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	R
	unknown_6:R
identityĸStatefulPartitionedCallˇ
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input

Ũ
#__inference__traced_restore_1527284
file_prefix2
 assignvariableop_dense_20_kernel:"@.
 assignvariableop_1_dense_20_bias:@5
"assignvariableop_2_dense_21_kernel:	@/
 assignvariableop_3_dense_21_bias:	6
"assignvariableop_4_dense_22_kernel:
/
 assignvariableop_5_dense_22_bias:	5
"assignvariableop_6_dense_23_kernel:	R.
 assignvariableop_7_dense_23_bias:R&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_20_kernel_m:"@6
(assignvariableop_18_adam_dense_20_bias_m:@=
*assignvariableop_19_adam_dense_21_kernel_m:	@7
(assignvariableop_20_adam_dense_21_bias_m:	>
*assignvariableop_21_adam_dense_22_kernel_m:
7
(assignvariableop_22_adam_dense_22_bias_m:	=
*assignvariableop_23_adam_dense_23_kernel_m:	R6
(assignvariableop_24_adam_dense_23_bias_m:R<
*assignvariableop_25_adam_dense_20_kernel_v:"@6
(assignvariableop_26_adam_dense_20_bias_v:@=
*assignvariableop_27_adam_dense_21_kernel_v:	@7
(assignvariableop_28_adam_dense_21_bias_v:	>
*assignvariableop_29_adam_dense_22_kernel_v:
7
(assignvariableop_30_adam_dense_22_bias_v:	=
*assignvariableop_31_adam_dense_23_kernel_v:	R6
(assignvariableop_32_adam_dense_23_bias_v:R
identity_34ĸAssignVariableOpĸAssignVariableOp_1ĸAssignVariableOp_10ĸAssignVariableOp_11ĸAssignVariableOp_12ĸAssignVariableOp_13ĸAssignVariableOp_14ĸAssignVariableOp_15ĸAssignVariableOp_16ĸAssignVariableOp_17ĸAssignVariableOp_18ĸAssignVariableOp_19ĸAssignVariableOp_2ĸAssignVariableOp_20ĸAssignVariableOp_21ĸAssignVariableOp_22ĸAssignVariableOp_23ĸAssignVariableOp_24ĸAssignVariableOp_25ĸAssignVariableOp_26ĸAssignVariableOp_27ĸAssignVariableOp_28ĸAssignVariableOp_29ĸAssignVariableOp_3ĸAssignVariableOp_30ĸAssignVariableOp_31ĸAssignVariableOp_32ĸAssignVariableOp_4ĸAssignVariableOp_5ĸAssignVariableOp_6ĸAssignVariableOp_7ĸAssignVariableOp_8ĸAssignVariableOp_9˛
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_20_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_20_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_21_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_21_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_22_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_22_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_23_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_23_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_20_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_20_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_21_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_21_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_22_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_22_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_23_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_23_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ĩ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ę

*__inference_dense_23_layer_call_fn_1527042

inputs
unknown:	R
	unknown_0:R
identityĸStatefulPartitionedCallŨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Û$
Æ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526941

inputs9
'dense_20_matmul_readvariableop_resource:"@6
(dense_20_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	@7
(dense_21_biasadd_readvariableop_resource:	;
'dense_22_matmul_readvariableop_resource:
7
(dense_22_biasadd_readvariableop_resource:	:
'dense_23_matmul_readvariableop_resource:	R6
(dense_23_biasadd_readvariableop_resource:R
identityĸdense_20/BiasAdd/ReadVariableOpĸdense_20/MatMul/ReadVariableOpĸdense_21/BiasAdd/ReadVariableOpĸdense_21/MatMul/ReadVariableOpĸdense_22/BiasAdd/ReadVariableOpĸdense_22/MatMul/ReadVariableOpĸdense_23/BiasAdd/ReadVariableOpĸdense_23/MatMul/ReadVariableOp
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:"@*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	R*
dtype0
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:R*
dtype0
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rh
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ri
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs


ö
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586

inputs0
matmul_readvariableop_resource:"@-
biasadd_readvariableop_resource:@
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
Ņ	
Ã
.__inference_sequential_5_layer_call_fn_1526888

inputs
unknown:"@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	R
	unknown_6:R
identityĸStatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
¤

ø
E__inference_dense_21_layer_call_and_return_conditional_losses_1527013

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs


I__inference_sequential_5_layer_call_and_return_conditional_losses_1526750

inputs"
dense_20_1526729:"@
dense_20_1526731:@#
dense_21_1526734:	@
dense_21_1526736:	$
dense_22_1526739:

dense_22_1526741:	#
dense_23_1526744:	R
dense_23_1526746:R
identityĸ dense_20/StatefulPartitionedCallĸ dense_21/StatefulPartitionedCallĸ dense_22/StatefulPartitionedCallĸ dense_23/StatefulPartitionedCallö
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_1526729dense_20_1526731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1526734dense_21_1526736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1526739dense_22_1526741*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1526744dense_23_1526746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
§F
Á
 __inference__traced_save_1527175
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop
savev2_const

identity_1ĸMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¯
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHą
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ą
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ũ
_input_shapesë
č: :"@:@:	@::
::	R:R: : : : : : : : : :"@:@:	@::
::	R:R:"@:@:	@::
::	R:R: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:"@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	R: 

_output_shapes
:R:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:"@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	R: 

_output_shapes
:R:$ 

_output_shapes

:"@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	R: !

_output_shapes
:R:"

_output_shapes
: 


I__inference_sequential_5_layer_call_and_return_conditional_losses_1526644

inputs"
dense_20_1526587:"@
dense_20_1526589:@#
dense_21_1526604:	@
dense_21_1526606:	$
dense_22_1526621:

dense_22_1526623:	#
dense_23_1526638:	R
dense_23_1526640:R
identityĸ dense_20/StatefulPartitionedCallĸ dense_21/StatefulPartitionedCallĸ dense_22/StatefulPartitionedCallĸ dense_23/StatefulPartitionedCallö
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_1526587dense_20_1526589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1526604dense_21_1526606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1526621dense_22_1526623*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1526638dense_23_1526640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs


ö
E__inference_dense_20_layer_call_and_return_conditional_losses_1526993

inputs0
matmul_readvariableop_resource:"@-
biasadd_readvariableop_resource:@
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
­

I__inference_sequential_5_layer_call_and_return_conditional_losses_1526838
dense_20_input"
dense_20_1526817:"@
dense_20_1526819:@#
dense_21_1526822:	@
dense_21_1526824:	$
dense_22_1526827:

dense_22_1526829:	#
dense_23_1526832:	R
dense_23_1526834:R
identityĸ dense_20/StatefulPartitionedCallĸ dense_21/StatefulPartitionedCallĸ dense_22/StatefulPartitionedCallĸ dense_23/StatefulPartitionedCallū
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_1526817dense_20_1526819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1526822dense_21_1526824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1526827dense_22_1526829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1526832dense_23_1526834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input
é	
Ë
.__inference_sequential_5_layer_call_fn_1526663
dense_20_input
unknown:"@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	R
	unknown_6:R
identityĸStatefulPartitionedCallˇ
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input
š	
Â
%__inference_signature_wrapper_1526867
dense_20_input
unknown:"@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	R
	unknown_6:R
identityĸStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1526568o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input
­

I__inference_sequential_5_layer_call_and_return_conditional_losses_1526814
dense_20_input"
dense_20_1526793:"@
dense_20_1526795:@#
dense_21_1526798:	@
dense_21_1526800:	$
dense_22_1526803:

dense_22_1526805:	#
dense_23_1526808:	R
dense_23_1526810:R
identityĸ dense_20/StatefulPartitionedCallĸ dense_21/StatefulPartitionedCallĸ dense_22/StatefulPartitionedCallĸ dense_23/StatefulPartitionedCallū
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_1526793dense_20_1526795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1526586
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1526798dense_21_1526800*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1526603
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1526803dense_22_1526805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1526808dense_23_1526810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙R*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input
Ĩ

÷
E__inference_dense_23_layer_call_and_return_conditional_losses_1526637

inputs1
matmul_readvariableop_resource:	R-
biasadd_readvariableop_resource:R
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	R*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:R*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨

ų
E__inference_dense_22_layer_call_and_return_conditional_losses_1526620

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Æ-
÷
"__inference__wrapped_model_1526568
dense_20_inputF
4sequential_5_dense_20_matmul_readvariableop_resource:"@C
5sequential_5_dense_20_biasadd_readvariableop_resource:@G
4sequential_5_dense_21_matmul_readvariableop_resource:	@D
5sequential_5_dense_21_biasadd_readvariableop_resource:	H
4sequential_5_dense_22_matmul_readvariableop_resource:
D
5sequential_5_dense_22_biasadd_readvariableop_resource:	G
4sequential_5_dense_23_matmul_readvariableop_resource:	RC
5sequential_5_dense_23_biasadd_readvariableop_resource:R
identityĸ,sequential_5/dense_20/BiasAdd/ReadVariableOpĸ+sequential_5/dense_20/MatMul/ReadVariableOpĸ,sequential_5/dense_21/BiasAdd/ReadVariableOpĸ+sequential_5/dense_21/MatMul/ReadVariableOpĸ,sequential_5/dense_22/BiasAdd/ReadVariableOpĸ+sequential_5/dense_22/MatMul/ReadVariableOpĸ,sequential_5/dense_23/BiasAdd/ReadVariableOpĸ+sequential_5/dense_23/MatMul/ReadVariableOp 
+sequential_5/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_20_matmul_readvariableop_resource*
_output_shapes

:"@*
dtype0
sequential_5/dense_20/MatMulMatMuldense_20_input3sequential_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
,sequential_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
sequential_5/dense_20/BiasAddBiasAdd&sequential_5/dense_20/MatMul:product:04sequential_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@|
sequential_5/dense_20/ReluRelu&sequential_5/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ą
+sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¸
sequential_5/dense_21/MatMulMatMul(sequential_5/dense_20/Relu:activations:03sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0š
sequential_5/dense_21/BiasAddBiasAdd&sequential_5/dense_21/MatMul:product:04sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}
sequential_5/dense_21/ReluRelu&sequential_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ĸ
+sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¸
sequential_5/dense_22/MatMulMatMul(sequential_5/dense_21/Relu:activations:03sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0š
sequential_5/dense_22/BiasAddBiasAdd&sequential_5/dense_22/MatMul:product:04sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}
sequential_5/dense_22/ReluRelu&sequential_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
+sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	R*
dtype0ˇ
sequential_5/dense_23/MatMulMatMul(sequential_5/dense_22/Relu:activations:03sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R
,sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:R*
dtype0¸
sequential_5/dense_23/BiasAddBiasAdd&sequential_5/dense_23/MatMul:product:04sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R
sequential_5/dense_23/SoftmaxSoftmax&sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rv
IdentityIdentity'sequential_5/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rē
NoOpNoOp-^sequential_5/dense_20/BiasAdd/ReadVariableOp,^sequential_5/dense_20/MatMul/ReadVariableOp-^sequential_5/dense_21/BiasAdd/ReadVariableOp,^sequential_5/dense_21/MatMul/ReadVariableOp-^sequential_5/dense_22/BiasAdd/ReadVariableOp,^sequential_5/dense_22/MatMul/ReadVariableOp-^sequential_5/dense_23/BiasAdd/ReadVariableOp,^sequential_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2\
,sequential_5/dense_20/BiasAdd/ReadVariableOp,sequential_5/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_20/MatMul/ReadVariableOp+sequential_5/dense_20/MatMul/ReadVariableOp2\
,sequential_5/dense_21/BiasAdd/ReadVariableOp,sequential_5/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_21/MatMul/ReadVariableOp+sequential_5/dense_21/MatMul/ReadVariableOp2\
,sequential_5/dense_22/BiasAdd/ReadVariableOp,sequential_5/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_22/MatMul/ReadVariableOp+sequential_5/dense_22/MatMul/ReadVariableOp2\
,sequential_5/dense_23/BiasAdd/ReadVariableOp,sequential_5/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_23/MatMul/ReadVariableOp+sequential_5/dense_23/MatMul/ReadVariableOp:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
(
_user_specified_namedense_20_input
Û$
Æ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526973

inputs9
'dense_20_matmul_readvariableop_resource:"@6
(dense_20_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	@7
(dense_21_biasadd_readvariableop_resource:	;
'dense_22_matmul_readvariableop_resource:
7
(dense_22_biasadd_readvariableop_resource:	:
'dense_23_matmul_readvariableop_resource:	R6
(dense_23_biasadd_readvariableop_resource:R
identityĸdense_20/BiasAdd/ReadVariableOpĸdense_20/MatMul/ReadVariableOpĸdense_21/BiasAdd/ReadVariableOpĸdense_21/MatMul/ReadVariableOpĸdense_22/BiasAdd/ReadVariableOpĸdense_22/MatMul/ReadVariableOpĸdense_23/BiasAdd/ReadVariableOpĸdense_23/MatMul/ReadVariableOp
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:"@*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	R*
dtype0
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:R*
dtype0
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rh
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ri
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RŌ
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙": : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
 
_user_specified_nameinputs
Ĩ

÷
E__inference_dense_23_layer_call_and_return_conditional_losses_1527053

inputs1
matmul_readvariableop_resource:	R-
biasadd_readvariableop_resource:R
identityĸBiasAdd/ReadVariableOpĸMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	R*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:R*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙RV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙R`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Rw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ĩ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*š
serving_defaultĨ
I
dense_20_input7
 serving_default_dense_20_input:0˙˙˙˙˙˙˙˙˙"<
dense_230
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙Rtensorflow/serving/predict:ö

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
ģ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ģ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ģ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
ģ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
í
3trace_0
4trace_1
5trace_2
6trace_32
.__inference_sequential_5_layer_call_fn_1526663
.__inference_sequential_5_layer_call_fn_1526888
.__inference_sequential_5_layer_call_fn_1526909
.__inference_sequential_5_layer_call_fn_1526790ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 z3trace_0z4trace_1z5trace_2z6trace_3
Ų
7trace_0
8trace_1
9trace_2
:trace_32î
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526941
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526973
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526814
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526838ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 z7trace_0z8trace_1z9trace_2z:trace_3
ÔBŅ
"__inference__wrapped_model_1526568dense_20_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ã
;iter

<beta_1

=beta_2
	>decay
?learning_ratemhmimjmk$ml%mm,mn-movpvqvrvs$vt%vu,vv-vw"
	optimizer
,
@serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Ftrace_02Ņ
*__inference_dense_20_layer_call_fn_1526982ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zFtrace_0

Gtrace_02ė
E__inference_dense_20_layer_call_and_return_conditional_losses_1526993ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zGtrace_0
!:"@2dense_20/kernel
:@2dense_20/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Mtrace_02Ņ
*__inference_dense_21_layer_call_fn_1527002ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zMtrace_0

Ntrace_02ė
E__inference_dense_21_layer_call_and_return_conditional_losses_1527013ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zNtrace_0
": 	@2dense_21/kernel
:2dense_21/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
î
Ttrace_02Ņ
*__inference_dense_22_layer_call_fn_1527022ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zTtrace_0

Utrace_02ė
E__inference_dense_22_layer_call_and_return_conditional_losses_1527033ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 zUtrace_0
#:!
2dense_22/kernel
:2dense_22/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
î
[trace_02Ņ
*__inference_dense_23_layer_call_fn_1527042ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 z[trace_0

\trace_02ė
E__inference_dense_23_layer_call_and_return_conditional_losses_1527053ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 z\trace_0
": 	R2dense_23/kernel
:R2dense_23/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_5_layer_call_fn_1526663dense_20_input"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
˙Bü
.__inference_sequential_5_layer_call_fn_1526888inputs"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
˙Bü
.__inference_sequential_5_layer_call_fn_1526909inputs"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
B
.__inference_sequential_5_layer_call_fn_1526790dense_20_input"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526941inputs"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526973inputs"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ĸB
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526814dense_20_input"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ĸB
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526838dense_20_input"ŋ
ļ˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ĶBĐ
%__inference_signature_wrapper_1526867dense_20_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŪBÛ
*__inference_dense_20_layer_call_fn_1526982inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ųBö
E__inference_dense_20_layer_call_and_return_conditional_losses_1526993inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŪBÛ
*__inference_dense_21_layer_call_fn_1527002inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ųBö
E__inference_dense_21_layer_call_and_return_conditional_losses_1527013inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŪBÛ
*__inference_dense_22_layer_call_fn_1527022inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ųBö
E__inference_dense_22_layer_call_and_return_conditional_losses_1527033inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŪBÛ
*__inference_dense_23_layer_call_fn_1527042inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
ųBö
E__inference_dense_23_layer_call_and_return_conditional_losses_1527053inputs"ĸ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsĒ *
 
N
_	variables
`	keras_api
	atotal
	bcount"
_tf_keras_metric
^
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs"
_tf_keras_metric
.
a0
b1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$"@2Adam/dense_20/kernel/m
 :@2Adam/dense_20/bias/m
':%	@2Adam/dense_21/kernel/m
!:2Adam/dense_21/bias/m
(:&
2Adam/dense_22/kernel/m
!:2Adam/dense_22/bias/m
':%	R2Adam/dense_23/kernel/m
 :R2Adam/dense_23/bias/m
&:$"@2Adam/dense_20/kernel/v
 :@2Adam/dense_20/bias/v
':%	@2Adam/dense_21/kernel/v
!:2Adam/dense_21/bias/v
(:&
2Adam/dense_22/kernel/v
!:2Adam/dense_22/bias/v
':%	R2Adam/dense_23/kernel/v
 :R2Adam/dense_23/bias/v
"__inference__wrapped_model_1526568x$%,-7ĸ4
-ĸ*
(%
dense_20_input˙˙˙˙˙˙˙˙˙"
Ē "3Ē0
.
dense_23"
dense_23˙˙˙˙˙˙˙˙˙RĨ
E__inference_dense_20_layer_call_and_return_conditional_losses_1526993\/ĸ,
%ĸ"
 
inputs˙˙˙˙˙˙˙˙˙"
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙@
 }
*__inference_dense_20_layer_call_fn_1526982O/ĸ,
%ĸ"
 
inputs˙˙˙˙˙˙˙˙˙"
Ē "˙˙˙˙˙˙˙˙˙@Ļ
E__inference_dense_21_layer_call_and_return_conditional_losses_1527013]/ĸ,
%ĸ"
 
inputs˙˙˙˙˙˙˙˙˙@
Ē "&ĸ#

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_dense_21_layer_call_fn_1527002P/ĸ,
%ĸ"
 
inputs˙˙˙˙˙˙˙˙˙@
Ē "˙˙˙˙˙˙˙˙˙§
E__inference_dense_22_layer_call_and_return_conditional_losses_1527033^$%0ĸ-
&ĸ#
!
inputs˙˙˙˙˙˙˙˙˙
Ē "&ĸ#

0˙˙˙˙˙˙˙˙˙
 
*__inference_dense_22_layer_call_fn_1527022Q$%0ĸ-
&ĸ#
!
inputs˙˙˙˙˙˙˙˙˙
Ē "˙˙˙˙˙˙˙˙˙Ļ
E__inference_dense_23_layer_call_and_return_conditional_losses_1527053],-0ĸ-
&ĸ#
!
inputs˙˙˙˙˙˙˙˙˙
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙R
 ~
*__inference_dense_23_layer_call_fn_1527042P,-0ĸ-
&ĸ#
!
inputs˙˙˙˙˙˙˙˙˙
Ē "˙˙˙˙˙˙˙˙˙Rŋ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526814r$%,-?ĸ<
5ĸ2
(%
dense_20_input˙˙˙˙˙˙˙˙˙"
p 

 
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙R
 ŋ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526838r$%,-?ĸ<
5ĸ2
(%
dense_20_input˙˙˙˙˙˙˙˙˙"
p

 
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙R
 ˇ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526941j$%,-7ĸ4
-ĸ*
 
inputs˙˙˙˙˙˙˙˙˙"
p 

 
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙R
 ˇ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1526973j$%,-7ĸ4
-ĸ*
 
inputs˙˙˙˙˙˙˙˙˙"
p

 
Ē "%ĸ"

0˙˙˙˙˙˙˙˙˙R
 
.__inference_sequential_5_layer_call_fn_1526663e$%,-?ĸ<
5ĸ2
(%
dense_20_input˙˙˙˙˙˙˙˙˙"
p 

 
Ē "˙˙˙˙˙˙˙˙˙R
.__inference_sequential_5_layer_call_fn_1526790e$%,-?ĸ<
5ĸ2
(%
dense_20_input˙˙˙˙˙˙˙˙˙"
p

 
Ē "˙˙˙˙˙˙˙˙˙R
.__inference_sequential_5_layer_call_fn_1526888]$%,-7ĸ4
-ĸ*
 
inputs˙˙˙˙˙˙˙˙˙"
p 

 
Ē "˙˙˙˙˙˙˙˙˙R
.__inference_sequential_5_layer_call_fn_1526909]$%,-7ĸ4
-ĸ*
 
inputs˙˙˙˙˙˙˙˙˙"
p

 
Ē "˙˙˙˙˙˙˙˙˙R´
%__inference_signature_wrapper_1526867$%,-IĸF
ĸ 
?Ē<
:
dense_20_input(%
dense_20_input˙˙˙˙˙˙˙˙˙""3Ē0
.
dense_23"
dense_23˙˙˙˙˙˙˙˙˙R