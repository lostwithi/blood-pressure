Ü
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
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������(*
shape:���������(
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
d
random_normal/shapeConst*
dtype0*
valueB"(   Q   *
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
Y
random_normal/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
seed2 *
dtype0*

seed *
_output_shapes

:(Q
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes

:(Q*
T0
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:(Q
|
Variable
VariableV2*
dtype0*
shared_name *
	container *
_output_shapes

:(Q*
shape
:(Q
�
Variable/AssignAssignVariablerandom_normal*
_class
loc:@Variable*
use_locking(*
_output_shapes

:(Q*
T0*
validate_shape(
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:(Q
f
random_normal_1/shapeConst*
dtype0*
valueB"   Q   *
_output_shapes
:
Y
random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_1/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2 *

seed *
T0*
_output_shapes

:Q*
dtype0
�
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes

:Q
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes

:Q
~

Variable_1
VariableV2*
shape
:Q*
	container *
dtype0*
shared_name *
_output_shapes

:Q
�
Variable_1/AssignAssign
Variable_1random_normal_1*
_class
loc:@Variable_1*
use_locking(*
_output_shapes

:Q*
T0*
validate_shape(
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes

:Q*
T0
f
random_normal_2/shapeConst*
dtype0*
valueB"Q      *
_output_shapes
:
Y
random_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_2/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
seed2 *
_output_shapes

:Q*

seed *
T0*
dtype0
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*
_output_shapes

:Q
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:Q*
T0
~

Variable_2
VariableV2*
shared_name *
_output_shapes

:Q*
	container *
shape
:Q*
dtype0
�
Variable_2/AssignAssign
Variable_2random_normal_2*
_class
loc:@Variable_2*
use_locking(*
_output_shapes

:Q*
T0*
validate_shape(
o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:Q
f
random_normal_3/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Y
random_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
T0*
seed2 *

seed *
dtype0*
_output_shapes

:
�
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0*
_output_shapes

:
j
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes

:*
T0
~

Variable_3
VariableV2*
dtype0*
shared_name *
	container *
_output_shapes

:*
shape
:
�
Variable_3/AssignAssign
Variable_3random_normal_3*
_class
loc:@Variable_3*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes

:*
T0
�f
Variable_4/initial_valueConst*
dtype0*�e
value�eB�e(Q"�eZ�d��/t�5Sf=�R�=�9J>�=�^��c�==8�9�����zf�<L�<����w~�<a莽Z�=�m�=e�=3u��l��>S8�>�z�;庿��n=2I��T�==��
>�&C>�?=�k��=�H=3��<���Q�=���r��9��üB���eH�ww�=������N�M�ϼ_:f=���KIf=��=3s=�O�A�����1��}=�F=��[���F�Qp ���>������i H�":��7ӌ=mIٻo�=ݥ�;ɂ=���=�罌	������&���K�_�s=�����>����8�ͽ'����ƽG��<�@h�˰*>�z���l3<�ѩ�G�6�m�=���ט>�H =�n�=�w�E4�<��=)>���<u��\�焠=`��=L�=�ǚ=�<C[=�`�<�k以j�Z ,>��k�:z�<Z�'>wt=�������;�3=ާH�KT1��)%=��f���_>]=�6>���=$y�q��Rͽ�8�='=@�|;=���;[c�8���b��b��g��4�Ľ���=s,��^-=�=:�l=7Y=������@��~�<���<�=+?=g\	>��+=��;��Y����jY!=�珽��=�l!>F
B>��>C��=��=���]��Hg=	���=�"���ؽ.]%=�t��t<)�r���HJ8=q�!>��`�/�ء#=���=�㽽m��=3���kf��/e1�C�=��=�Sa��M=�J8=�b�=�E=V@K=ز��[�4���<��뼏�o�z�<�%N=$�<u�=��7>�0���c���_=".�=����_�����=��<3��=�eн����T���>��ޡ=؀��k�=e3+�VZ>�*�L°<+�=C� >X>����;��=��q<�=�İ�B>s`q=[��=*,׼��R�<=:�<cd��'�v>g�B�̽
> �<�f��M?>�Ci=��;�*}=�Do=[������ٽ�+�=���Ѵ=�����>�O���c�=��>���
����o���t"�Tr�=�-���ż�갽?p�=Ow�=4���t>��U��=�kҽA3=snn�Ϗ>C���Hͽ�{ҽ`�b=�T\;;ች�����=z�e�xY��\E>)XD<"�=��N���=5H��2SM��{��z?���=�B����<�{=w�g<-��=�2����<ǑI��#`�h.�=xG�����򵔽���G�=�:J=	�<p|x<��B���2>��@=��=��=�Y���>������=�z ��E�e��<zJX�ȷ���7=C�;5�%>;�=���=�4>t--=-&W=0T:\>��,E�=~>c��l�
����l�<H�=�]�=�t�=�нW<7�ݼT �� 	�=_�=���=��>˷�<�Ƚ_%�<�*1�fI�w�ѽRʲ=@��s�,=�=:��=�����<��l>��b;T0H�G�<�1ֽX �=;�6`=���=7���f�<Ǝż7������S��=#/=w�=}4<=KA��3O�=-^�</l���F��(���b������=�@���<�,�$<C=���=��>���=ߎ5������X;<�]�����"�ἒ֘�;e��6�=d�������9N=�Zn���s��~=0Pi=
U���(��L���74�#.�=�2P=LDK�#g���,�=�x�=�Z��b�)>%�7=�&�=�b�Û��#1��j�"=�0����o[�|>���=+�`����oV���p^��>�o���!�=��R��� � ٗ=7H��5K>IV�=�+)=�>�����<&)�;(��=pr<o��R:+=�3�O/�<�'�~L'��=_r���������r3�m=�<BI8�+3>�=�+��������=���=E�+=Sz =C��=hP|�W�n=x>Ѽa�=*�v<:�K<����~����,�[7���"<��(�;�����=��<�u=�"��7g�=��ʽܮ�`O꽿%q=��&�k���:�M�%=���Q�ȋ� �*���ǽ[f��6�����<�n������=�O�;Jؔ={ǚ=>��Ƚ��ۼ}�>�U=�">��;�}�[m>}9�;��_P=���<���='먼-��=�=#��=���/]�<6��=��	>Eq�=�3q�{�8>��=��z=#Q>�Qy��I>'n%���e�>��s=���=�F���<[�f����#��=(:�=Bo�'�Խ0?W��<˽]�h�ǃ:=+?罠s�:�~-��\��>f ���ɽ��=��=����
�=�fM�+i= <{ �=%�r>���8��aH>����0=#↽#��=���=B��r�ͼ.��=ۦ�%�T��`�5�X>d�C=XrV=���>�������t�����IA��p =��s=*u<���<�
��V=�e����'/����o�� =Q�$>�@�=(z��O�J���i=R���	<���fr>��b����y4���<8����rr=ҵ�/5�<�=�2��v�=͙X��n�`��=���Rz�gg�:%��=c��=�z�<�b=��>?Cm=��<M^�<w���׆�=�Ҽ��{=}���`��<>H��u�<߳��f=iv��ߞ=w�y���߻��ս�������S�<�w�A��=��r<�8ؽ��%���=�ڽZ*"��gʼ��`=�R<�f/>w�5��}8��]�%�	>�c�<@���������=����iF���e=��h���9C@�5�ǽ2���r��7D9=���=��?=K'�/����=�)=�I����{j"����='9h=���_W�=<9�겲=��=�g���<�I(>�JT>�����@;����,�􀥼򥝽�����WK���!>͌�;SF?<��#>m�m=�ѽ Hܽ���=�Vw��LO>E�o�E>z���X4=��H�3�o=��=�_&>����O=HK����n�U��'�~���߼Xp�=�Z�<(x�: �_>!Ɠ=�!�=C<+�w��=S)>X���m��,���Eٽ��</2<�P�<��>��2>k&���;�������o>�k�<� �=�/!=���B�=���=��={����Z;���=F���=$_��wE��)�<���8��=��+<�qT<#ܙ=v���P}=�$�==V>�`�= Z>�a<��߽�󰽥��;x���HEs��2>��=�q =%3�{�s�=��$�n]ƽw��?�Q���)<�n½��A�?�}=�I4���y<���ʢ�=���Z��<[��=���/Ѽ+�==J���Mk�M�_�GO�4##>����>��U���i=[a�=�������<le%�����8;���)�j/t��Q��{����ս J
=R`n��ᏽ=Z�����<pn�=��!=/��;=Z⳽o/��$J��F@��;���g��=!����g�<�=]6�=7�:Ɉ��R}=��۽%4��{�E� �`����<���<j�=��c=�P.�71P����=Wق��m�E���:�<��$���/� 1=_,>bn뽠�q�m	�zH�=C��<�"��O��<�=2Z<L�>=�����'�ے>6�5��?�<�A!���e=�:��;�=��Z=�����5 ��)���H�<�F�@�i��]¼?f��ou�<�x�󨡽���=�ؘ�Z,s�U�^=����}=�8�Ӂ�=Nಾ��_=�������+����h>�[��J�;v�<�|��(�ý?>4>�?7=����R�O�on&��j=�>�={�>�:�=��Z=��p=�î���=�6=��� ���2��=�fo=��;�H>X��=Iv�=���=9��:�!b��]�=���=�K����=ze>R��6n��o�>��;��z_��G->Gjȼc ȼC꛽�65��A]>'��u�S=�\F>�;=�L�=V/>-�>���N��=���<��������4>�G�=���;��� ,�<�*`�~8=��T��"<�YN!=�T�=�> �:=
ҹ=�3�=ܸ;;�Ž%�w<k�+<��Ĭ�=���:k=%��<P,o>�c7>�
=�ǚ=�/�=���<����;>>��G�m>+<<��>k�>k6y=�:������p��z˽{��<���J�:���8+����=c�μ��?=S.<>�p��é��tE=g'�=��7���=)����=�y�=�
�:���=��=Gm�=�)��y��欨����7<C��=�!���]�jrX;~������/I�=��>�=�_�=��=���h��<���=�i��{�<med�'rǽM貽�V;�$��}P=G�*�cI�=�����I�;�TU=C���L=���=X�<���:�:x={�Y;<�E=Z���#�c�#�.=�q��
1����T�,�=�� ��w����=g<�u��=����6;��7U���;��%���Z<��=����� �=��&�[�=�4�<�����ǂ�=�8N=Ä'���=���<O�=�X����	>?��=��?��&���O(�;�<>��C�$��v�=���
(��䝽p�:�eKa=��I>]�����<����h{>K���ˎ=����"��<���=�"<SLG>M�~=#�=��=ܶǽB ��R5~�{]$��$�hỻ�u޽FW�=�f��m�6�d9=k�=�J>�B��L���	�ɽ*<�z=�qp��m>��;E>��ڽ�>F����<�mӼs=�L�;����4�'S>�e�l��?�=u��=�3>=��>�I�� m��=�u<��=t�Ž��ü����B=o�4�g��=���<��<bB߽ZnL���W=��F=����Mݽ������F<ϴ�<o8>�� ����䗖� �=p����C�=�=�<{�>�O�"��\S�g"�=C�>���=�8�9jt����<BOQ��h[=�%t>�u��:�^��d#��������<H��9=���䛆=�v=��ѽ+!<�cܼ{�=�����=�vӽ��>>���=���n�=0|�=�n�=���jqY�R�[<���=��=$=���9�=��#= w="�f>�k�=�Z� 73=�:��{�ٽ�����|��ڽ� >8Rؽ���l����>R":=� ;���ɼ����������=�̊=������½�_�=Rd��>ǽ=%������)�=��˽�4�=����:<wv:>a���� �;u�S�9=Є̻���_n�g>>�)�=��ݽk:����=K�)��"�<]�P�y�=�vϽw�'>W�$���O>c҃���#�KG�=C>�]̼w�~>����I�r�L=�.
=�V�����<�L�=d
��h��=@tؽ��=���t�A��6�����<0)Q=�2b��&=�Q����<�ļ�>>h�U�=���A�<m�K�%���>������=�.��K�<�W�=��<����<�k#�T�1�g�P��og=�#�=&�=c�V��o�=���=���;������=U�<��R<�P�=.�=XҼ3C)>ς�=( �<@E�=��=d�����S���Sb�==�<�Л�K�	��/6�4=;̙=$�
=��G��yּB�ս��� ��2=3�j����=\1��F�G�e=�Ġ;8�J=b�;>^��<�t�ﾽ��H=wY�=��<%I����:��	>u^c;�_��#-=��>��[�hZ>?h�<:rp<Q����=}���w)=e8= A���Dp��:Ӽ�=?�|=� ���=`~���;���T���&>�?<<`�}��׼%�H<�q�=(�l�.�=Qʻ<c���e�k�=��>��4�ȴ=h7�5�κ�k���Խ�2�=z�=d3J=���C�<l >���H�\<'a�=���S��=�
>�u�%��<#�ؽ����n�=�=��B�<GwQ���2�c�'=����;\����S�½w0�;Kf��v���F�=E��=��Ǽh_>ȉ���ݽMA6<;�)���ȯ�=�H�<��2>�.;��½I�
�����>���6˻�C�=߰f8{�=%���Z�����=�(�4�aw���립��*��7�;��f�����=s���{�=QF��"=G�'�9�6<P��=�M�<� ��3����.o=��<ܐ���)"���9�[��=��<�=�>5�<�C�;�ٽ���<] ���ۼD��<=ýmm���{>�u�=�0D�O0���M=R�"�7|�,d����E?���C<:Ѽ�@=/�����@��Y���>��Ľs3�O�+>�M=��<o�P�� *>KO���=��#>�s>K�=��_"���l=dg>���<��=�
;�S���.Q=��,>��=7ҙ�}�=���=�=��<���=�{l>��*�9=ܑ.�;��=]B;�p+��<���y� ����˼=�z�=÷��0L�3c���=R�@=�=ο'=����k}�=^,�߱A�RZE=�ɇ�>��=��ٽ3�o<*��=)�#>�eN�ӣL=U�3��O��zT���e��kD�Sy<� >#�6>����꺽���;�Ž��2<s��<9D�<��&��=����������J�<��9����2W�= ����{��R����=�T�=�d=��A�#�����g�0<�DZ<Ǫ/�+��<8��ӵ��?�<Є�= p�=��/>�E=�%�=w���">��r>�k>b��s'"�f�=��?<�_ҽ���='�<��>�[�$=}���ft2��bF=��=���=�D0��	y��<���;�&��%�O��C�bV�8�=ꩢ<�s��E�����=���s�= p���$>��K�I(��ֿ�	�=�E�=�5>�����b#>a���;|g��e�ͼ���<ch���&(��=s �;�Q򽨗 >����=s(��u��=xh�����=gx��P�U>c47>��ཪ��=i�'bJ=ꮺ���=�Zm��]=��=EΣ=J����>�w�=sF�����=�S���b=;Pֽ#�G>�=�+��c��=&)A=W��=w�
>�zq����>�}��=�<{Ϧ��ۺ[+�'�o��=���<?�ʼC��=��=J|-���/=�����PKW�+:=�U��MB>�8�=����ήH�<��=e>%f����[�u��;�3N���ڽj7>[Y)�3�?�%�ѽ�'6�\?=�>:->U�=H�ڽ�^k���� w�!�<?ґ;�D��Hc�=��<��K��k>���=GT >����C ��˽S89=�h~=����
\�"!#>8�>)�=�v��H�򵽖��=��=�8�=�v(�9>�B�����<���=j��:?>�6�=� <�>'��m}��Z���c�=Ť<�,�=Δ��6U>��=���=���i����!!�:�� ]=��9*g�=�&�&/�;I��=U=���Ȩ�=�H�A 4;ࢽv�>�	>/��]�<NҰ=Gk�<E���F��=��h=ה����
��\�>Ȥ�=I�:i�='������
�]=G6>�`�Lk�/����U=o_,>�4>�j�w=9*�����z=�DI>��=C뉽�Y�=:K�<���fӫ<�4>ǡ��A�G�h��⼝N&=@l=��<!�PA�=7����<_޻=����>`��>/�޽e!�=�Mp�L���������v��=���=��X�'4>o��=a��!S��7�=�2����=Uܼ[gr>- ��zR���a=��L;��+�z��=�Aϼ8���2�=ð�=B�3��==0]��W�=�(��3��{U>'�>k>��.=~ZD>�Y�=<񣽗��"�c�%�l=w���=��u��{�t+���*�z��;�p�=d=c(����<�L��PE߼��T��5��)�3>��b>�>�<=�1���nϽPr��wt���1�>�=��=�ֆ=���rgݽaV=�=mu�*�~��=�j�= ��=�-�=�]�B���/2>�k��>o�� ߻�� �=`�;��->�7x=<���/R�(�l��,�<�1>˷�<�\�=�L�<�$�=�ƿ=��u�����J3��l�=��G<C,�=.�=��>w��<*s>Ce�s4�=�t���o�=�?=��~<Wy<�?>��`���0����h�:=�7��+x�=�9�=G��:*6�=X�f=6�>�h=��_<B�μ���U���1��ȧV>#��=v*�����NB����[�=S�2����+��<$6,�b�B�����=(�7=c;���=��O=W��
�= �꼖=Mrx��W����J={�Z:/b>���WB�'6��^9����=�#?�W<� s�=Sۺ�I��=��<+҆=�c�m�޻�j�=��L�+7<?t>s�ؽ�^�<�z>n>��%�����N��=�<�
�#���/��<xjŹ�c����=B�� �ϼ�}
>۽>�3���>2fE<+�����=ңĽ���Z��M�<$G=�kc�=�n<mi�=/��=�.�=�L�HR�=2^��r�����=V�=��>-4n�e�̽�\O;3\N�#��=f� ����=|;9�W�=��)>��,��
���<Q$?�(醾�9�Äy=�{=�6�=d�>'�,�Vi�<�J�=iE>�ꉽ�l⼋��<O�R��jm>q�=?�	�!��=h����k�=[�=�2�<[a�������=����׫����;|�]�\<]���@���t>�|��)�|�H�W=x=��<�랼
=?c{� ����>����L�= �>���=�ő��G�_��=�f��k�=[׽,���3⽓:ﻹ���4>s=_>o>��>��&=0�z�:ێ�{_��"��̍�rx�����J�x�>�9��a���B=7��=M"�;�zW�/h6���&=�i�����='��e=����o�Y����ǥ=�¥<oa���˪9��Q�=ǄƽtV��$��=�SA==��=U��94*��E�ږ�=���=��Ͻ�ȥ�o�U=x>H�j����, >u�>'RR<��=L> �����=���<b�`�̜½o�8=�9˽��=å�=@�ͼ`��	i��W�='�G=^�����Ӻ��= �p<��~>�E=��<�=�
��P�=�-O=�.�#L=d��=���=��Q=�����E�n�=_ƌ���m�#u	>�͕<Ș�=��K��~<N"�� h���?q>�uc�/���)6>Q7�=m��mWI='L�;D�c�5���.���k>V�=��<���=1�=�[+��'�=�X/=w�	��-C=�	�������>���=�l���&��ZO>"5��1l�W�+>kü�Fu=��=׻=�齸�m���v=�  =�٫�������b�<3S���>��a�5��a�$������Y�=w�|=4j�C=>O��͊<�����8���2���<�=�3�p���^��&ʽI٨<��'>}졽ڢ���������='X<�<=���=�>'���߂ӽK�);���<�
�=�B;43�<r�����$>�伞�g�=	뫽�x<�3>�_k>�4�cu���d��i̼E������\4>�ԯ���%>�aܽ�Ͻ)5=ˬu�D*����W���<�"���5�-> ��ʏ���Y��Ӳ��l�<<碥=��=)�<��4����>��Ƽ=���d�=�%'>mc�*gM�
�,��=���X��=J{��o<o=�ԛ<��6<��=� l��}�D��kV��L����:������Sμ�'2���н��=�m�<�*��%���=:��ލ2����;/d���H�'=0e��g�==�N�>�=U�ڽgp��dC=�%$��a�=�>�5+M�,�-�q����d��k >��6>F�>r ~��wU���V���8����=���<��=�<�h̽C�.<'�b'��{�+=��=9��=ec�=��^>3q?���<s�=(�[�i}<�_�� �	3�=� 9���jī�����">�ϫ�0��=��=������<E+�=N;6�[��=mq�=���;:�=T��=�I���eQ�`y漮�>��珧9��μ����k>���=P����+*=�	����4���g�h�+���;#���1@2>O��=P�o��>��I�#>w�=h�м�ʳ��X)=�ң�
�w=J�=�Y;\��M�
>c6=�cn�i9�=CF�N
>)�=���=��ý9{�=�w��=�����5G�=E�j=#�1>Zw�{�>��"�w�7=�c>����>�<�!����
=5=�+�����{�&>�r��Cۏ<R¥=�v�=��_� ��6ҽ�!$=-5�\$7���U�\�=���=�	�=`U>_x�=�~\��][�O�R��/���<��ֻz��=��=`��:�%u�.)>�ѽP�m�K�"=8֮<<�r4�7	>g2���a�;�=���3�;}_�UK�<]ʽ�>W>c��;~:=�;S=�Χ<��<�^=>��>���=��6='j�%�D��ָ���X<=�<zV�=?��<B���S�Խ�޺ϼ@��5���>Sؐ;�&�=C����wE�f=�<o
m=v:K��*��W�w�=��P�պԘ�?�ѽױ�1c>w���i�>�Fj����a�>Om��VT>�V><'�=|b���>y=c�<�޽Ԣ�:��=�Ƽ�b���F>a�.<��<�/>WOQ���-��<�P�W��<�v&>�>8>�=ї��t�M�=���=#�N���׼����Ma7�7-�=�l����<�$����J��� ��ߐ=`�=� ��Q ���=a~F���>�k�<� >���ӈ=�>K&><?�b�Y��{���� 6��G�>�/׽Rmn�/��04��P�>��>��C��T=�ͽjA=A8�Y�=7��;���=Ǐt�'{��;��=�R>>]Y=hx:"k�C��=?�U�=�l=@�Žw�"=d9L��w�=�5�=�Dȼ���@f�<XD�=�0��� >7$��E<�;��= ���`�<:F���j���´;�d����<z����Ľr�ֻvA��O��R+�c�=��O=��=���=%�2>J6Y��r.�M�>5:�(��=��<�����0��(�k ƽ{��tq;=Ǎ�����=�����_=4�
������<E��a����,=���=�z �b�=~A�;���d�=�1=�3@=��.=�6��Oe=���G��y�9;9�=d�=��P=7���t�=���=�c��@:=�>zz=����M=�ʹ=�DѼ�y�=n��S�pH>�=�*�=J���RVݼ�ި=)n��.�RS�=h�=KU��0�$� {�<@�ܽ�3���E>;+H�#��=T�=	d��RTM�"b!>��>=[0��=��U�@�q���̻���s��=ς'>�?��=��<ܵ>���=;�=��>_G߽���=!��`�Q>\���Î=��k�_t�'`�d¸<���=n`ý&c >G춽�=�=+���=�=G@�=���q<�C=�tϼ� ��u<L���)��=�G>�=Z�xK=ٰ�����%��O��`W=��O>'>�(x;�i����ע�<R��2`>s&G=�Q<S����^;@(��Pn=�FĽgHn�3��uY�:��<:��=�O�6>�cE�1d��ڽk��o�=#�=ԓ=��=j�>��M�2Q�ͽO��=�`$>��޼~W��"ׂ����箨=��ȽO?*<y1�=݋8=@�J<u��pT�d�˼2�=�0>:����5���7=3#r:ә�=4�=����2��)P=��,>"�F����ao�>�L�;��O=B�H�%������xN�=9����=�<�=_k=.�"�����>Б�=_\h=/"��Q��T�< ����=���=�����͋>��=ہ�<:�=;=Mx��XS�=8�=�Ͻ��c@��>$��!~�&$>q������=�N>~?��ZF���<�������C�=�6�=�ӿ=�'�;6��sl��%𶽅� >[R�h�}���;=��<�� ��t�3�=�lս���=�Ї;Jfw��j�:E>_6���=�M(>5�ƽ� H>FF̼����d��c��<����V<�V�x2�=�<�7��<G�=V/�=�[���E=�1.�C3�<E�U<�L>�	>$�=�h�=?a)�[{5=}@�=7��=�K��#��}�#=�w���-�Mcm�Zg�=ua�=���;���c��^�-�u>�B�<S)�=�6�=���_p>'��=Wi/�8�>����Q[8��L���6>K�K=I�;�ĩ=��ܽ�@н���G+���<=W}=�p��:=i4$������y6�=��$>�'Y�w�->cJ���;��P���=%��;
��=H==�&=�ļ��=�Y�<s�,�& ������EB=O�5��?�=�>5j^���>��;�O^>BN6���7�B�8��-<�)$�S��=}�=����E�ӽs�м�m!�K�g=�
��E>��;���<���=KL�=T,��`�!����<�V�<��<� �;�q�<��;S�����=�WR	��F>�Zǽ�C�=��JM�;��KQ6����=@M9>��5���v;�;��D;*�%�)���(3�=�y(=�0|=�N�p��;�?��/�[v���+K�Pu��`2���>oGS=
���^|J=y]J=%��:u�>9��>�=��f����=�=��*=�c�W���ڗ$��s>��Z=��=*
_output_shapes

:(Q
~

Variable_4
VariableV2*
shape
:(Q*
dtype0*
	container *
shared_name *
_output_shapes

:(Q
�
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:(Q*
validate_shape(
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:(Q
�
Variable_5/initial_valueConst*�
value�B�Q"��ƃ=����l��qƼD7��~h<�p�W\�=�f=�&���@�<0 S=x�b=ȉc�B���]����a�#+�J���T��_�W��X��ߠ=��=�w�={��\�=�3E>����夽��<X�K�ۼ���=��	>C��<����Aqºc��<= �8�s_4=��ٽ? =�U�=oM ��5i�DF���=2{�y����>&�I>:�v�y�>�c�;��;W>S�{ځ������o[=7���A�=Ia�=����D�=ꑽm+��FT<6
�<H&d�J���/ě=�J��wq��p&��-������=`�,<8Y��*
_output_shapes

:Q*
dtype0
~

Variable_5
VariableV2*
_output_shapes

:Q*
shared_name *
	container *
shape
:Q*
dtype0
�
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
use_locking(*
_output_shapes

:Q*
T0*
validate_shape(*
_class
loc:@Variable_5
o
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes

:Q*
_class
loc:@Variable_5
�
Variable_6/initial_valueConst*�
value�B�Q"��?4=�ۅ��`���r��=�g9���q>8tm=XƼo��>�>�8�|=�/�6����)t=��;���n�<-1�=_Yl=cP'��-ҽg��=���=��>��=o^@���	>��3�7�ƽ;Ј����B>��,=ə��4o��FY=�U����>=j=,��>�3�=>�=�kཧq�=����7=)I=P������'|�=�)�=?(�=�Y>�'u=�b�=5�e���>Ϋ���K^=�Ļ��=C�8��;�!�<��>&�E=�5�>��<�h�����=�L#>�^�=�{f��&>l0���ʽ{���ѫ:=��*
_output_shapes

:Q*
dtype0
~

Variable_6
VariableV2*
shared_name *
dtype0*
	container *
shape
:Q*
_output_shapes

:Q
�
Variable_6/AssignAssign
Variable_6Variable_6/initial_value*
use_locking(*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:Q*
T0
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:Q*
T0*
_class
loc:@Variable_6
m
Variable_7/initial_valueConst*
dtype0*
valueB*���*
_output_shapes

:
~

Variable_7
VariableV2*
shared_name *
dtype0*
	container *
shape
:*
_output_shapes

:
�
Variable_7/AssignAssign
Variable_7Variable_7/initial_value*
use_locking(*
_output_shapes

:*
T0*
validate_shape(*
_class
loc:@Variable_7
o
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes

:*
_class
loc:@Variable_7
�
MatMulMatMulPlaceholderVariable_4/read*'
_output_shapes
:���������Q*
T0*
transpose_b( *
transpose_a( 
U
addAddMatMulVariable_5/read*
T0*'
_output_shapes
:���������Q
C
ReluReluadd*
T0*'
_output_shapes
:���������Q
�
MatMul_1MatMulReluVariable_6/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
Y
add_1AddMatMul_1Variable_7/read*
T0*'
_output_shapes
:���������
G
Relu_1Reluadd_1*
T0*'
_output_shapes
:���������
�
$mean_squared_error/SquaredDifferenceSquaredDifferenceRelu_1Placeholder_1*
T0*'
_output_shapes
:���������
t
/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
_output_shapes
:*
out_type0*
T0
u
3mean_squared_error/assert_broadcastable/values/rankConst*
dtype0*
value	B :*
_output_shapes
: 
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
�
mean_squared_error/Cast/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/Cast/x*'
_output_shapes
:���������*
T0
�
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
_output_shapes
:*
dtype0
�
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
$mean_squared_error/num_present/EqualEqualmean_squared_error/Cast/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
�
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
.mean_squared_error/num_present/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: *

index_type0
�
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
_output_shapes
: *
T0
�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
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
:���������*
T0*

index_type0
�
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*'
_output_shapes
:���������*
T0
�
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
�
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
mean_squared_error/Const_1ConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB *
dtype0
�
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

mean_squared_error/valueDivNoNanmean_squared_error/Sum_1mean_squared_error/num_present*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
_output_shapes
: *
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
p
-gradients/mean_squared_error/value_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
r
/gradients/mean_squared_error/value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
=gradients/mean_squared_error/value_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/mean_squared_error/value_grad/Shape/gradients/mean_squared_error/value_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/mean_squared_error/value_grad/div_no_nanDivNoNangradients/Fillmean_squared_error/num_present*
_output_shapes
: *
T0
�
+gradients/mean_squared_error/value_grad/SumSum2gradients/mean_squared_error/value_grad/div_no_nan=gradients/mean_squared_error/value_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
�
/gradients/mean_squared_error/value_grad/ReshapeReshape+gradients/mean_squared_error/value_grad/Sum-gradients/mean_squared_error/value_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
4gradients/mean_squared_error/value_grad/div_no_nan_2DivNoNan4gradients/mean_squared_error/value_grad/div_no_nan_1mean_squared_error/num_present*
_output_shapes
: *
T0
�
+gradients/mean_squared_error/value_grad/mulMulgradients/Fill4gradients/mean_squared_error/value_grad/div_no_nan_2*
_output_shapes
: *
T0
�
-gradients/mean_squared_error/value_grad/Sum_1Sum+gradients/mean_squared_error/value_grad/mul?gradients/mean_squared_error/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
1gradients/mean_squared_error/value_grad/Reshape_1Reshape-gradients/mean_squared_error/value_grad/Sum_1/gradients/mean_squared_error/value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/value_grad/Reshape2^gradients/mean_squared_error/value_grad/Reshape_1
�
@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/value_grad/Reshape9^gradients/mean_squared_error/value_grad/tuple/group_deps*
_output_shapes
: *B
_class8
64loc:@gradients/mean_squared_error/value_grad/Reshape*
T0
�
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/value_grad/Reshape_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
_output_shapes
: *
T0*D
_class:
86loc:@gradients/mean_squared_error/value_grad/Reshape_1
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients/mean_squared_error/value_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
_output_shapes
: *
T0
�
3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
T0*
_output_shapes
:*
out_type0
�
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*

Tmultiples0*'
_output_shapes
:���������*
T0
�
+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
_output_shapes
:*
out_type0*
T0
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
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
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*'
_output_shapes
:���������*
T0
�
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 

9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapeRelu_1*
_output_shapes
:*
T0*
out_type0
�
;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
out_type0*
T0
�
Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
valueB
 *   @*
_output_shapes
: *
dtype0
�
7gradients/mean_squared_error/SquaredDifference_grad/MulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
7gradients/mean_squared_error/SquaredDifference_grad/subSubRelu_1Placeholder_1?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/Mul7gradients/mean_squared_error/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
�
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*
T0
�
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*
T0
�
gradients/Relu_1_grad/ReluGradReluGradLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:���������
b
gradients/add_1_grad/ShapeShapeMatMul_1*
_output_shapes
:*
T0*
out_type0
m
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0
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
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_6/read*'
_output_shapes
:���������Q*
transpose_a( *
T0*
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
_output_shapes

:Q*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������Q*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*
_output_shapes

:Q*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������Q*
T0
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
out_type0*
T0
k
gradients/add_grad/Shape_1Const*
dtype0*
valueB"   Q   *
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������Q
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes

:Q*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������Q*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes

:Q*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *'
_output_shapes
:���������(*
T0
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:(Q*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������(
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:(Q
b
GradientDescent/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<
�
6GradientDescent/update_Variable_4/ApplyGradientDescentApplyGradientDescent
Variable_4GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:(Q*
_class
loc:@Variable_4*
T0*
use_locking( 
�
6GradientDescent/update_Variable_5/ApplyGradientDescentApplyGradientDescent
Variable_5GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_output_shapes

:Q*
_class
loc:@Variable_5*
use_locking( *
T0
�
6GradientDescent/update_Variable_6/ApplyGradientDescentApplyGradientDescent
Variable_6GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:Q*
_class
loc:@Variable_6*
use_locking( *
T0
�
6GradientDescent/update_Variable_7/ApplyGradientDescentApplyGradientDescent
Variable_7GradientDescent/learning_rate/gradients/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:*
use_locking( *
_class
loc:@Variable_7
�
GradientDescentNoOp7^GradientDescent/update_Variable_4/ApplyGradientDescent7^GradientDescent/update_Variable_5/ApplyGradientDescent7^GradientDescent/update_Variable_6/ApplyGradientDescent7^GradientDescent/update_Variable_7/ApplyGradientDescent
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_1d33a4a3a3734e8ea875f188aa96820e/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*q
valuehBfBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7*
_output_shapes
:
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
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*q
valuehBfBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7
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
loc:@Variable*
validate_shape(*
_output_shapes

:(Q*
use_locking(*
T0
�
save/Assign_1Assign
Variable_1save/RestoreV2:1*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

:Q*
T0*
use_locking(
�
save/Assign_2Assign
Variable_2save/RestoreV2:2*
_output_shapes

:Q*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
�
save/Assign_3Assign
Variable_3save/RestoreV2:3*
_class
loc:@Variable_3*
_output_shapes

:*
T0*
use_locking(*
validate_shape(
�
save/Assign_4Assign
Variable_4save/RestoreV2:4*
validate_shape(*
_output_shapes

:(Q*
_class
loc:@Variable_4*
T0*
use_locking(
�
save/Assign_5Assign
Variable_5save/RestoreV2:5*
_class
loc:@Variable_5*
_output_shapes

:Q*
T0*
use_locking(*
validate_shape(
�
save/Assign_6Assign
Variable_6save/RestoreV2:6*
_class
loc:@Variable_6*
_output_shapes

:Q*
T0*
use_locking(*
validate_shape(
�
save/Assign_7Assign
Variable_7save/RestoreV2:7*
_class
loc:@Variable_7*
_output_shapes

:*
T0*
use_locking(*
validate_shape(
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
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
GradientDescent"�
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
Variable_7:0Variable_7/AssignVariable_7/read:02Variable_7/initial_value:08*�
serving_defaultv
-
input$
Placeholder:0���������()
output
Relu_1:0���������tensorflow/serving/predict