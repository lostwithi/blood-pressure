��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.14.02unknown��
n
PlaceholderPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
d
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   9   
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*

seed *
seed2 *
T0*
_output_shapes

:9*
dtype0
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes

:9
d
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes

:9*
T0
|
Variable
VariableV2*
	container *
_output_shapes

:9*
dtype0*
shared_name *
shape
:9
�
Variable/AssignAssignVariablerandom_normal*
_class
loc:@Variable*
T0*
_output_shapes

:9*
validate_shape(*
use_locking(
i
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes

:9
f
random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"   9   
Y
random_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2 *
T0*
_output_shapes

:9*
dtype0*

seed 
�
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes

:9
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes

:9*
T0
~

Variable_1
VariableV2*
_output_shapes

:9*
	container *
shared_name *
shape
:9*
dtype0
�
Variable_1/AssignAssign
Variable_1random_normal_1*
_output_shapes

:9*
use_locking(*
T0*
validate_shape(*
_class
loc:@Variable_1
o
Variable_1/readIdentity
Variable_1*
_output_shapes

:9*
T0*
_class
loc:@Variable_1
f
random_normal_2/shapeConst*
_output_shapes
:*
valueB"9      *
dtype0
Y
random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_2/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*

seed *
seed2 *
T0*
_output_shapes

:9*
dtype0
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*
_output_shapes

:9
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:9*
T0
~

Variable_2
VariableV2*
shared_name *
_output_shapes

:9*
shape
:9*
dtype0*
	container 
�
Variable_2/AssignAssign
Variable_2random_normal_2*
validate_shape(*
T0*
_class
loc:@Variable_2*
_output_shapes

:9*
use_locking(
o
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes

:9
f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
Y
random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*

seed *
seed2 *
T0*
dtype0*
_output_shapes

:
�
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes

:*
T0
j
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes

:*
T0
~

Variable_3
VariableV2*
shape
:*
shared_name *
_output_shapes

:*
	container *
dtype0
�
Variable_3/AssignAssign
Variable_3random_normal_3*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes

:
o
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes

:
�2
Variable_4/initial_valueConst*�2
value�2B�19"�1f�����`.�jR���R�d�>:�Ľ��^=�M���= ��:s�=�Y��Hz=�h���O8=CY^=.{:��=�b<��=��I����������=�2	=b��=�]=E�=��=��=�(���� T<��Y�G<7=�����v�;_��=_7�		>[����;=CЕ��V�<h��<��*�:+��$��=w�ƽ����=[ad��"�=�00��Z>�9J>�����<
����Q�<Aد;�ƛ�P.�v�
>��b��?W=�����=؆� ��;b��<>� >�k���^=�@m=rz�=��=�����=�E���=2M�<���<D#���*I�����U1�����%=R3>��A=(�w=��=�R�=yɼ�� #��<wm�=n8&�;�
���"8<�G>��8;�K�=�A=�9
=�=Z�>g�C�r�ݻ4�>�>F��<�
��n������u��uT������}Ž��B=��ƽ��	>]X0:,�#=�t=���������˽B�<'9�;�1�=���=X�t=G|,<���<���=�-S>�<�=��\��;��*�����=E>ؼ@�u=K\�8{7��>s�����=�-0��1���=�,�_�νC��;A�e��<a�,>7�$��<]�����o�X
��J�����<��=��R=o4����̽���e �m!�=j]�=@���]6�=�bԽH@�����<�';=��:>#�0=���<b������;Y�<��A�=6>� �;�E)�Ex���;��f=B.�=q�@=O�7�?�=LI����� �=�ʖ=�k���=OCg��9e=d��7��+/p=I�A�Ž���=����-��=G�ٽ-��=X���0)���; >�=�K=NC�DP��P��=Mb�:=�=ӃD��=��=�M=��W<������=��>�����R�<�,>v�>�i���ԃ=�^��l���I�=-N��0M=�J��)��=, �=*�e=��=�u��x=�7���F�=��.�_�6=��(=�',=ޥ�=�o=�C>H&׽�\�=3������(�W��k�o+r�b4=ߜ�=L��<�}A>>i*���>$����,�3�0=�!�=f9����<�����������=��>Su�=B[�����r�=fl<�vS�و�s�M�{�=�n�=��/=Y��=;� >m��<�M�=�#���2�GYս��=m�=ݰ=OƤ�=ꄼ{p�W>�@��{�<֭�����C<�{���"d��GL>��<E��=�����UW����=��۽*/�ܵ3>�,�?\�[��:�E�=�N|���O�-��*��x�=�1���D���>�ܣ�'��=��=�Ө=U� � o=�	�Hp=�Sӽ3q�=���=U�<w�>�=o������<�ۙ���>q>7'<�7X�=��=�=��`��%+����=%j˾+�&<�J�K <��t�k	� ﶼ�Q�<���=#;��>S���I�=�
���=��.>K~�<�>�
��6{�=��
>�l�xb���zbx�r��;�����y���7��!��=����;ޫ=s�<���=���>mzݽp�!=s�^��cͽ@��<?�b=sڽY������1�=�
ĽҼY��<E�>�O�C�Y�)�3K;>��<a!>` Z��WC�P��[��s�R�G��=����> ����<� ��+/׽C>���<��=�t�=���=�ޙ<�t >aG�2�=/�+=�&����=�q�=�\��� �=�|ƽ�� �n�����'��<�b>+�/�[%��*w~���A=��=�|�=:"&='�g=�h�����U���<=�,��S��Ұ��]J��椽�qT=���M"J���b�'��=sUk=�����r�=3���V/<�7>�}=Hq<C��=����!c>ν'�=2�1=揋=M�>(��<�0��`cB;�j�?��=�$>p��a���{?��[�=x�%>Û
>�dH��j=+�=Ҟ�=�>m	�����=��>\<㘲<�@���|�>�cN=����_a=nNýJE��׫��뫞�U.�=󦀽cI�8zt=�t���ƼD�.=L>�H+>�ی>�)캼z?���a=�Q�x�$Շ����<cgD<�0�<tM;�V=�$E>X���@>���>Gv��plC�aX>�^��`��7���\�G�ފý�~�[�<�ȹ�M�+=M�S�>۽�=�fY>���e�L>��̽���<8hm=�D->�3}<�*�X����Q�P ̽.�=�ܤ�"g<��$>1%��c��Ou�����F{�p��o�C��,>;��;Q,=�Ѱ����=8�=�b�;�P��iH�j���:�=�����н�ǌ=m�`^g���>}�T=w�=�
�栽��x= ��1>�r�=�L�c��t���:"��N9>��u=Dd>1		��4�����Q=��ϼ�[�ȋ�؜=��e�w�Y=�91�V�=xlK���k�Z��,� =��=����x����;�x���e�5�<�gY<75�=X�k�5���W�=E���Jܽ��;O&u�?�>K)ٽ��\=h��9M��=��=O>F>S��=rP#>��=��=4�4�?��5A��>�{D��{=��j=fn=�#ܼO6>s9;�"
��匽<�(��E�����=�F%��q��{��<�f,>�qC����=1��=R��Sc������=t,���ǽ�.Ƚƀ�(�>�4b��~��]#�=0�i�?��=LY,��ZU��z=��>bT��H�ҽ-�;�^o<5"�;ܽOpf���=�����<�I�=�0�<�+��-�����X��{��;�� ��?
���<��>�$D>��=V����&>���=׍�<�9��Wz=���=�cR>N�E>�N>�S�>��>�|��QDý=��=��5�V��[�m�o�!����Jv=(���yϽÊ&�;�����_Cr��1�����J�3��T�=�`�ufb>(�l�\ �GÈ=|��%�=X�=|�*>�^P���E>�c>�6o���T=�s=��� >ҡ&��d=��>o��=�#_���?��b�����=E�"�=r�	����=h70=��e�Ĩ��C8ټ¨7�'d=C=�ԃ��)�j��}�=��=/�v=�y���'�>l ����;*|	>���U6���>Z�<�D=#��=c�=��轆p>k�H=3*�	v;�_�+�긷�>�X_>��]<Z<7�=�����dT�S8�����: a�<�p(���>�Q���=\v�=���@S��g@>;$L��IN<�_�\�7>(�j{ٽB�����SE�Z�w�3�&<�
��4J=x��#�=;C�9��<J�<��n>M�-�I\�{>��<��=�j<�	s=>G��<[�������=OZ:�;�ߛ>�p8����=��=7��=屢=����;���#��<|�7;���6\="5�=�.<B|μ��7�<��~>�'=xG��w���!;�)����=�����ή�b�����%��=!>%ʨ���ƽ��:=���<#�(�<k������=�+��C�A=��=EY>o��=���*)��L�z&���ü=;:���=��C�`�=�o��']*<C���G�7�[.�=+�}�\p4="�=5�=ԭ
�D��=��*D��Ͼ�o�Ǽ��S�Q=��c�):=�N+;����Ɲ8��ET=�?��R�E<S(3�(%>��<�:5=��>�X�=T⹽`HG;k��U�=�jW=�\�=��U=t�=�gyн���R��Pjf>�����ܽ2[�hCu���輼bC>nu�=��R��!�L̰���<9�=#��=#�;�M�>���4t�=�ܜ�)=�"=}.=����j.�2��=a=�M>7��2�z<�=���E=��d�c�٤�Ϫ!�6�>b��<�\{=�����6�=��u=xr�3սK��;�B�=�������-�=�#��p>��*=��`�������׾������ҧl���Լ�&��1�>5����V��e�=n�&����$]>�}� ��edw<X�
����k�<���=�
�}�<f�����g=�#�=�Ǎ�!�>k=%>�Q�U�W;���<{ϱ=F��"�ooi��S��X��v��C��>����������ˌ� ={�=,�<w��r����Z�T=�����)>�>5���C�_��=��T=S��=�B>�&Z��4�'1=���<��X�}[{�#���:��=;�b�t�=�����&=��ռ��R�>�5�<	�2>����߆��z��$��3�+<�"��e��=o��<�= �=�LG�đ�(�g�X����9=�.���t;i�=��=���;��9>w���ך�<+O��!��p	=Խ�e#>B�e�z(��[����'�D�I>緽�i�w��=��D=c#��UF�[����P���������I|=���)Ƃ�j�/��>�=P�!�M�O:�2��KE�͖=�#������e`�<�&��U,�/�ƽ���=A)�����m=X������k��$�>*�5=|>�';<����,>�V}�,������<��Ͻ��۽$��<C����=��='�>��߽3��=���+��=��h��,������->�*���s�=�����ּ�}A=��]��N�=/���<��<��_�O�s�tt>C}H�S'B>>��=���<�<�
�=�k��0i���^��0��K��<�c�;���=���=B\߽�ƀ=#a�<d��Y��=]�]����<�>��8>S�E�c�����GI7>0�=7N�<���<T��HM�\q==n�%�><�g���>?��<~��H��<§2�WaԼ:lڼt\*���E>���[<�{����=�ս;tZ����*Ӿ=��<u�|<2p��;	=���=�wD=Vt2���ɽ����^	>c��=��<�٩��F5�a�5�7��=x���c�<3���y��?D�Ч���J�w�`�#A��_�
�"�N=gJ �,����z��{>�=�ʼs�U���_=�]4;9̎>}��=X��=��=���w��=	a������_#��E�=�Xd=٤��!��<�y�X�<�r��6�=��=�8=��:�"/���ƽ֭��.׽�,(�p�>��=i�*<�)���=�;$.>(�=?x>�˨�}�=�>��P>$�#�h	�<����Gey� ���݅��0�<�[=H�s��l;��=�>>��B>���=-�=�I��ձ��-F=V�����2�L����:�&
��[˽Ks�����\���Gu�=�7=;@�<�ǽ���=��=��=O�����|=z���`\�=��Ͻ;��g�߼�ݽ�t��[�=��O,�����ey�}([>v<<#��T򡽑,��e�=��a���d={��}�k�e,�<f`���9�' d�xV����������}����'>],0=m+�=����� >��M=Z"2��v���>V(����</|��b V;]=�<�<S�1�d(>�]�һ_=C~�=�+˽f��=���=���ׯ=S��<�P,=oX�<�/��	�=zQ��R���3�+"��k�[=�3�G|>WLC<2
�;]�=+ھ�3�ŻJW�<���=��>Y<�M���wؼ;��=���=��Q=GE�?�ļ��B��"��'�=�w�<j�Q��{����H�=.�=��	>�޷��5!�g�>-�ݽ�c|=���=:>�U=Sf�=�5=����øݼ�E<3LX�@�n��!����=�]'=ր��s��L�=���=->�|>c�l=��K�o����UKY=��*=[/j=�Ѽo��=[��@N���9������r:˽?<5>��9�}4�=![���<A>��+=K�>jZ�=�u��S�>���=ޒ#;r@���/=�I>��s<!�=� ���۬�s>�=g�� �<G�Q�tG,>���;Yb��ŭd�7Cܽ�>`�w�����=��=w�=G��
���jֻo}���,>�N�;j�ϥ=[ȥ��	1��&�<�[>s-k���r�e`��R>�j_�'u:�hdp=l1�=�$%��E����;�u
>��<�o\=X�>��X��T�pfJ���潨3�=�c\>t�����cɽUC���!�=��缐��<�p!��Ž!�@�!ů=����Ϫ��]��轵VF>���;���U�m=���<�[�=e�S���|����=�.�%�ϽRY߼�=��þ=��j��%ͽ�d�=n�S!�>�B>�'?��l=*
_output_shapes

:9*
dtype0
~

Variable_4
VariableV2*
shape
:9*
	container *
shared_name *
_output_shapes

:9*
dtype0
�
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
_output_shapes

:9*
_class
loc:@Variable_4*
use_locking(*
T0*
validate_shape(
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:9
�
Variable_5/initial_valueConst*
dtype0*
_output_shapes

:9*�
value�B�9"��� :�-�<� �h.x<�����p�=�^S> ����罤���*>e ~�[LἿ{U���=�(нSp�2� ��~�=:_���4ٽxJG���T��$����}>�/����
��6~��<��'!�<�q<�Qs;|���O޽�L��/��cB>�U�����=�XV����e�=kp���4t��� ��7��R!�{>ug&=I&�� 5�=`��<t�=瘈���̽����
~

Variable_5
VariableV2*
shared_name *
shape
:9*
dtype0*
_output_shapes

:9*
	container 
�
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
_class
loc:@Variable_5*
_output_shapes

:9*
use_locking(*
validate_shape(*
T0
o
Variable_5/readIdentity
Variable_5*
_output_shapes

:9*
T0*
_class
loc:@Variable_5
�
Variable_6/initial_valueConst*
dtype0*�
value�B�9"�Qդ���#��u;��;XF�=�v>�2��Cu=YI��8=�y:�%���=�o=#�����K=#� =������<��>��8=��Ӽ��=#ed�m罛���B�Ͷ�=2�G="����,,=���<[���S��~H>���C���皻_S.>���=�v�=5�R�B��=#��;�̤<�߼=<|<ڊ�s�=����~"�k�����<@�ѽ����e�=*
_output_shapes

:9
~

Variable_6
VariableV2*
shared_name *
	container *
dtype0*
_output_shapes

:9*
shape
:9
�
Variable_6/AssignAssign
Variable_6Variable_6/initial_value*
_class
loc:@Variable_6*
_output_shapes

:9*
use_locking(*
validate_shape(*
T0
o
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
_output_shapes

:9*
T0
m
Variable_7/initial_valueConst*
dtype0*
valueB*}ͺ�*
_output_shapes

:
~

Variable_7
VariableV2*
shared_name *
shape
:*
dtype0*
_output_shapes

:*
	container 
�
Variable_7/AssignAssign
Variable_7Variable_7/initial_value*
_class
loc:@Variable_7*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
o
Variable_7/readIdentity
Variable_7*
_output_shapes

:*
T0*
_class
loc:@Variable_7
�
MatMulMatMulPlaceholderVariable_4/read*
transpose_b( *
T0*'
_output_shapes
:���������9*
transpose_a( 
U
addAddMatMulVariable_5/read*
T0*'
_output_shapes
:���������9
C
ReluReluadd*
T0*'
_output_shapes
:���������9
�
MatMul_1MatMulReluVariable_6/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
Y
add_1AddMatMul_1Variable_7/read*
T0*'
_output_shapes
:���������
N
dense4_outputReluadd_1*
T0*'
_output_shapes
:���������
�
$mean_squared_error/SquaredDifferenceSquaredDifferencedense4_outputPlaceholder_1*'
_output_shapes
:���������*
T0
t
/mean_squared_error/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
�
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
u
3mean_squared_error/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
�
mean_squared_error/Cast/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/Cast/x*'
_output_shapes
:���������*
T0
�
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB"       
�
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
�
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
$mean_squared_error/num_present/EqualEqualmean_squared_error/Cast/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
�
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
.mean_squared_error/num_present/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
�
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B : *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B :*
_output_shapes
: 
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*'
_output_shapes
:���������*

index_type0*
T0
�
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
valueB"       *
dtype0
�
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
�
mean_squared_error/Const_1ConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 

mean_squared_error/valueDivNoNanmean_squared_error/Sum_1mean_squared_error/num_present*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
X
gradients/grad_ys_0Const*
valueB
 *  �?*
_output_shapes
: *
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
p
-gradients/mean_squared_error/value_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
r
/gradients/mean_squared_error/value_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
=gradients/mean_squared_error/value_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/mean_squared_error/value_grad/Shape/gradients/mean_squared_error/value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/mean_squared_error/value_grad/div_no_nanDivNoNangradients/Fillmean_squared_error/num_present*
_output_shapes
: *
T0
�
+gradients/mean_squared_error/value_grad/SumSum2gradients/mean_squared_error/value_grad/div_no_nan=gradients/mean_squared_error/value_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
�
/gradients/mean_squared_error/value_grad/ReshapeReshape+gradients/mean_squared_error/value_grad/Sum-gradients/mean_squared_error/value_grad/Shape*
Tshape0*
_output_shapes
: *
T0
m
+gradients/mean_squared_error/value_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
�
4gradients/mean_squared_error/value_grad/div_no_nan_1DivNoNan+gradients/mean_squared_error/value_grad/Negmean_squared_error/num_present*
T0*
_output_shapes
: 
�
4gradients/mean_squared_error/value_grad/div_no_nan_2DivNoNan4gradients/mean_squared_error/value_grad/div_no_nan_1mean_squared_error/num_present*
T0*
_output_shapes
: 
�
+gradients/mean_squared_error/value_grad/mulMulgradients/Fill4gradients/mean_squared_error/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
�
-gradients/mean_squared_error/value_grad/Sum_1Sum+gradients/mean_squared_error/value_grad/mul?gradients/mean_squared_error/value_grad/BroadcastGradientArgs:1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
1gradients/mean_squared_error/value_grad/Reshape_1Reshape-gradients/mean_squared_error/value_grad/Sum_1/gradients/mean_squared_error/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/value_grad/Reshape2^gradients/mean_squared_error/value_grad/Reshape_1
�
@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/value_grad/Reshape9^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/value_grad/Reshape*
_output_shapes
: 
�
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/value_grad/Reshape_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mean_squared_error/value_grad/Reshape_1*
_output_shapes
: 
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients/mean_squared_error/value_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
dtype0*
valueB *
_output_shapes
: 
�
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*
_output_shapes
: *
T0*

Tmultiples0
�
3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
_output_shapes
:*
out_type0*
T0
�
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/Cast/x*'
_output_shapes
:���������*
T0
�
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
�
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
�
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*'
_output_shapes
:���������*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*
T0
�
@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
_output_shapes
: *B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
T0
�
9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapedense4_output*
T0*
out_type0*
_output_shapes
:
�
;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
out_type0*
T0
�
Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
_output_shapes
: *
valueB
 *   @*
dtype0
�
7gradients/mean_squared_error/SquaredDifference_grad/MulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
7gradients/mean_squared_error/SquaredDifference_grad/subSubdense4_outputPlaceholder_1?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/Mul7gradients/mean_squared_error/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
�
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
Tshape0*
T0
�
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
�
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
�
%gradients/dense4_output_grad/ReluGradReluGradLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencydense4_output*
T0*'
_output_shapes
:���������
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
m
gradients/add_1_grad/Shape_1Const*
valueB"      *
_output_shapes
:*
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSum%gradients/dense4_output_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_1_grad/Sum_1Sum%gradients/dense4_output_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes

:*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������9
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:9
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������9
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:9*
T0
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������9*
T0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
k
gradients/add_grad/Shape_1Const*
dtype0*
valueB"   9   *
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������9
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes

:9*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������9
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes

:9
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:9
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:9*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
b
GradientDescent/learning_rateConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
�
6GradientDescent/update_Variable_4/ApplyGradientDescentApplyGradientDescent
Variable_4GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
_output_shapes

:9*
use_locking( *
T0
�
6GradientDescent/update_Variable_5/ApplyGradientDescentApplyGradientDescent
Variable_5GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_class
loc:@Variable_5*
_output_shapes

:9*
use_locking( *
T0
�
6GradientDescent/update_Variable_6/ApplyGradientDescentApplyGradientDescent
Variable_6GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
T0*
use_locking( *
_output_shapes

:9
�
6GradientDescent/update_Variable_7/ApplyGradientDescentApplyGradientDescent
Variable_7GradientDescent/learning_rate/gradients/add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
T0*
_output_shapes

:*
use_locking( 
�
GradientDescentNoOp7^GradientDescent/update_Variable_4/ApplyGradientDescent7^GradientDescent/update_Variable_5/ApplyGradientDescent7^GradientDescent/update_Variable_6/ApplyGradientDescent7^GradientDescent/update_Variable_7/ApplyGradientDescent
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign
Y
save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_ae33f5f74c204535b8ee07046cdb3bf1/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*q
valuehBfBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*#
valueBB B B B B B B B *
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7"/device:CPU:0*
dtypes

2
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
_output_shapes
:*
N
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*q
valuehBfBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7*
dtype0
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*#
valueBB B B B B B B B *
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2
�
save/AssignAssignVariablesave/RestoreV2*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(*
_output_shapes

:9
�
save/Assign_1Assign
Variable_1save/RestoreV2:1*
_output_shapes

:9*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_1
�
save/Assign_2Assign
Variable_2save/RestoreV2:2*
_class
loc:@Variable_2*
validate_shape(*
T0*
use_locking(*
_output_shapes

:9
�
save/Assign_3Assign
Variable_3save/RestoreV2:3*
_output_shapes

:*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_3
�
save/Assign_4Assign
Variable_4save/RestoreV2:4*
validate_shape(*
T0*
_output_shapes

:9*
use_locking(*
_class
loc:@Variable_4
�
save/Assign_5Assign
Variable_5save/RestoreV2:5*
_class
loc:@Variable_5*
T0*
validate_shape(*
use_locking(*
_output_shapes

:9
�
save/Assign_6Assign
Variable_6save/RestoreV2:6*
_output_shapes

:9*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_6
�
save/Assign_7Assign
Variable_7save/RestoreV2:7*
_class
loc:@Variable_7*
_output_shapes

:*
use_locking(*
T0*
validate_shape(
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
	variables��
A

Variable:0Variable/AssignVariable/read:02random_normal:08
I
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:08
I
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:08
I
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:08
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
R
Variable_5:0Variable_5/AssignVariable_5/read:02Variable_5/initial_value:08
R
Variable_6:0Variable_6/AssignVariable_6/read:02Variable_6/initial_value:08
R
Variable_7:0Variable_7/AssignVariable_7/read:02Variable_7/initial_value:08"�
trainable_variables��
A

Variable:0Variable/AssignVariable/read:02random_normal:08
I
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:08
I
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:08
I
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:08
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
R
Variable_5:0Variable_5/AssignVariable_5/read:02Variable_5/initial_value:08
R
Variable_6:0Variable_6/AssignVariable_6/read:02Variable_6/initial_value:08
R
Variable_7:0Variable_7/AssignVariable_7/read:02Variable_7/initial_value:08"(
losses

mean_squared_error/value:0"
train_op

GradientDescent*�
serving_default}
-
input$
Placeholder:0���������0
output&
dense4_output:0���������tensorflow/serving/predict