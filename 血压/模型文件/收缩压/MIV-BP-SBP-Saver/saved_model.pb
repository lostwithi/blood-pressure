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
PlaceholderPlaceholder*
shape:���������!*
dtype0*'
_output_shapes
:���������!
p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
d
random_normal/shapeConst*
dtype0*
valueB"!   C   *
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
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed2 *
dtype0*

seed *
_output_shapes

:!C*
T0
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes

:!C*
T0
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:!C
|
Variable
VariableV2*
	container *
_output_shapes

:!C*
dtype0*
shape
:!C*
shared_name 
�
Variable/AssignAssignVariablerandom_normal*
_output_shapes

:!C*
use_locking(*
validate_shape(*
_class
loc:@Variable*
T0
i
Variable/readIdentityVariable*
_output_shapes

:!C*
_class
loc:@Variable*
T0
f
random_normal_1/shapeConst*
valueB"   C   *
_output_shapes
:*
dtype0
Y
random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2 *
T0*

seed *
_output_shapes

:C*
dtype0
�
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes

:C
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes

:C
~

Variable_1
VariableV2*
_output_shapes

:C*
shape
:C*
dtype0*
shared_name *
	container 
�
Variable_1/AssignAssign
Variable_1random_normal_1*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes

:C
o
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes

:C*
_class
loc:@Variable_1
f
random_normal_2/shapeConst*
dtype0*
valueB"C      *
_output_shapes
:
Y
random_normal_2/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
seed2 *
T0*
dtype0*
_output_shapes

:C*

seed 
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes

:C*
T0
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*
_output_shapes

:C
~

Variable_2
VariableV2*
shape
:C*
shared_name *
	container *
_output_shapes

:C*
dtype0
�
Variable_2/AssignAssign
Variable_2random_normal_2*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*
_output_shapes

:C
o
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes

:C
f
random_normal_3/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
Y
random_normal_3/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
seed2 *
T0*
dtype0*

seed *
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
_output_shapes

:*
shape
:*
dtype0*
shared_name *
	container 
�
Variable_3/AssignAssign
Variable_3random_normal_3*
_class
loc:@Variable_3*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes

:*
T0
�E
Variable_4/initial_valueConst*�E
value�EB�E!C"�E��<��*X'=
�z�!	>B�;km%�z[>:�S=�껽%���ĩ=aR���<(�(�==�$<uX��y��L���zE�(�߽p=C�M>C�3;QE�=��=��j�rk�_J�=�-�.$��X<��8�~�K�i=z�������HʽKI6=+�*=�6:�� =�Yu��?<�\B��)�=n�=U9�<w��=ys>�Җ=��=�U >2��=t���<<�ei��S%�PI>�U�=��<sa���5=ɉ<#�n<Sj�E�d;��=���=�K!��z�Ӄ7�m	w�k��=�Ͻ{C�kGm=�=~����=7�ļ �=$HH=r�f��`��τ��Gc>�.z<��\���=����M�>=�T_=��=h���������� Ƽ�����[��=;��=�~
=�ĥ>�bh�B��@s�=�%<jn�<u�_�O��[Ӯ=��Z��#=q1L�V۵�oȤ��W=�E=�$�;�넽G�<����Jٽ��!�Y�?��U����=�м��H�bR��@,>/l���"�;�	ǽ�N�=��>���=�:.>w��V>��M>]f<X��<� v=;(���̼՘<�A5=�Q�=f_�;�n˽��=����
>M�=;'��HFg;���ݣ��	���\=":�<
ꬽZн�^5>kC��:5&��0$=�;(��4�W�&:�$�c�!=��=�3l��-v=�l�=�=��	ĵ�k��.7<*:b=��JS8��vF>X�`>�cH=��p5ҽ�I";��<'� >~�3=��/s^=`RC>W��O�P�����C��=ȕR=V0�;�^<���=��>��;z�8�W�>GWq=�p���;���=�����?�=�䛻[��=� W=�sȽ�wؽi�&>�E�=N��v�=��j� �=�~D�oe/>�,>�<>�'o=�1K���=CV����<�/�W&��E >R7��5R>��{<7��<RX�=�{>���<�>�#�ƽ&�>�8�=3|��@�=�50��D�=�=�;�='��:��k=#�>=p���P:>�ɥ=����=E��<ZA�=����L��5��M�5w�:"�Ľ�x�=M��=K+"����;��.=��=+����*>_廨��:��<{Uҽ�B�=#l+�<M/����<H��='�)�(�N=Oc�<��=Q�=��6=ӻ���W�_J=]�V�2\��>&�=��U; ����ED�M������=;<�zs��e�Խ�k=]A�K�b��K>3������0�;g��<���<��<�@���6=ͷC�B&��O=�V~='��<#?l<\����=���<�M�=��=�*Y=�����Pp=�a�=@F��}zx��α��r�=�m�<��t�'RY�W��<�g6��YU=����չ< G��/�
>�l��Cv�<�1P>4�=�}�O��=��=���<
�w������]��rm���I>�޼�=���=�F컒]2�+ts=u��=�0$>ւ��-��Ǖ����:�Fd�=�r��6�Kٸ<�	�=��=��齋�D�{Υ�X��=��};�xƽ¾7�-��Ӽ�=94K��4�ٵ*=���=�b>��>�������=�ͽ�&N>�<q��H���Q�'�����Q4<(@R��_:=���=p�;=(���Pg�An>�t`<�I���>n`B>�ub<�*>O�>=aI�<�8>�Ce=ݽx=e#��T��j1u=CX.�~ް�_l<	S$����#=W��q�=�8P>BGؽca-;���.f=_��<H�s<���<mt�ٴ��NW�>S�&=jj�ȬI�K��=x?~=�RO�z���,#=-(���h=���;?��=�8���^U���=��K={* �=`�<<񧈼c
3<�O=L�����<���K�y�̽(�н����S�>��,�=�=�s�.�ʽ�䎽�i'����=�T>ي����=��=蝃�f�=��=F����>�<���O�@=Ⱦx���>G U�o�R>7�����=D�ƽ�L�=T�=A�8<��<=���=��$��A���½����8�=7Q ��?>� =��h�3�;� ίʴ==���j	�k�O>]�<S�3>���<%�t�����pЮ<�\>��J��jU�C1����<�VؽnT >b�=*^�=�����=J#Z=���.�=��=b�N=;�E�鰍=�ύ=�mM>L61>�a���(i�{@?=�62���=.�����=7yL�n_G��:���j>}B�OMX>�R@���<��=�}?=�^����x�cme=F���R�=MXC>c��� >WT=��"���;��=���<<M6�3W뻻^?���=AA,�O>���=QL<��I>?���D""�*�r�#��=on&�����G�<�P�<B�N<���=�&�Q�$>��$�u��Hu ����;M�<m�غ�˄��]�=�$=pٶ=;s4>�E.>@�콥Y��(ȃ<�˰�}&�=rz���{=ix���WR=#��=8������<x�=�f����=�xĽhxr;�34>��=lI�={�r=�	�S$�<M<>��Q���=3���og=Z�,��`:=_���O�=���l"�ˤ�=C=>�g�=Ҍ��"�\��t���~�9�=r>��g>����bV��l^�=������=�����=��>��=�G=�u=wc���;���<l��=3`���iJ����u>S�<ޝ����<6ۿ��|��ܼ�<�J�=�߼T#��	s�=+!C=�z>�S�C�=~v��'+�)d�����=l9�<~zĽ`)�<>�8=�� >p��<+X���'��_Gͽ���=o^=@˴��έ�������=�н����C�<�K�ө9��>�S�=�%>�Q�=�����>���M�ջ��Ƚ�dP=��V�HKp=J1�<��?�x�2�0X۽���<�ۯ=eף�ǃ�=�Z����?΃=��C�_>��=�\>���=��ٽ�`��s�/�=b9>�\�=�B�}>�<�0�=����c�j>�Ѽ���
�>� X�*�+Tm�QU=�|g���f=��ۼI2��N=�>W��=6�C�:��]�=T� �P�="�>�=��5>��B;�t�=�	 �;���kҽUW=d��=
т�~�?�5��=6�$=�u,=��p>���<��S=m�q�ES<'v=�Ѣ=�
>�4�&�=���O =���=�Y�=���G=��	�KQ�=ya��G@<>�N_<x��:o�>�>�Ύ��\󒽐�<gGE�����)n=�s$=��D>��<K�ֽ2U><#@ս8->wPw���V=���<��&>�댼�����J<�=��n<:�"�8�.=�6�=c<��=M����ɋ��Ha��X@<�<+��˦�=o��͔�=���<���=o/���܈�C��pK���'�s6�=h>�
��D�������<>l��E>a>�<8�]���=�C=�g�h�~���2���W���>�6�<
��=��=�=����b���'��~��WU�*�}��]�<��l�� ���	���=�h���dY<��>��C���X�w?.��W�=�̙=37�=��%���<�Ľg�w�m�B=X1>u3�<K�=�=+��L�=k����;�)H��λ���=��=���=g4�=�<o�5��d�����I�=Ԓ�=��׼�U�=H�����4��=��&>�8�>�碼[��>̚���K�� ���Mz�=�
��@�ˮ���h��x�>B�S=w3C>?�T��j8�ֽ�;?E1=�$��ū�� >GT�=������\:���Ü���Y1=[���4'�'�=M�P=�+
� k��b�V=}���������<�_�0�Z>�bϽ�K��h �L��^ט�������н�\���2��(�ͼKF(������ƽ�P�K���5�M>S`�p.B���ֽ�r=��M>k-*>� <k�]='St��d%=BԒ=E���S�~���r�'+'>�t=��o�k�=��ǻ�L�����=�V>�A��J(o=OA)>���Ⱥ����P��@>�矽� =�Tu<6D��?K �<5��3�=H�V>K��=�2^�BM�TD��Ǯ���[[='�;�#��#��U��;<�˼[�>������v-�q�+�@�=��;�T;x�k�O�w>h7$�+�<��=ֲ�=k��=��Q=� �=	������=�y��W㽪:�<Y,5=W��=�^>pA���/�?�;�n��<0�z�G�>�u����� ��=�g�*�M����j=�N�=�䅽��8��v��Yڼ#�#<���<H
������ϧ�=)�=���=Sr=BG��<�=ZV�<%%c���"=C[�@#�=�摽*��=3੽��=-=�t>۫!=�a�a��=Ӭ#�k�}�_�V<O�R���E��/���; nκ�(�;#X�����{�7m>�!�=xl�=	`�%�=zw�=�I�=%�6=���/��<���A�6=��p==>Z���=3�SD1�'��[�<{p>�=�<_��=|x0����=U�/��(�s$N=��{=Ek�=k컽�;���˻O^�<�|�=V�=���������m�=�Z8=�8���>���_�=�5�ڄm��·<�4.���=�B����=�>IGD�	����1�=ng�=f�c��=��)=��o=:�Ө��z �= �>�`,O>�'�3��<��W��">�;x�=�ٸ�˛�;��мkp�<��>B��=�G=���=�=l��=8EZ=Y>��=5�����W��:[�=����kC4>�-�=�� >J׀�K�>O�>�<j��:�0�K@@�x�f>g"��*��=��v�m��<��o=��k=C�%�Z��=��?>� i��A�p,�<t�=Z44�P+���=��+=�U��#R<��c����Ĥ>��O�>�=�:+�3�i����6��.Z�;+��>��c=w� ;w�=T�F�7I�1�6>L�¼�ɽ�q�=�ȝ��o�(�>}J�=�4>#�;~DȽ�K���;�)>���Z5���&�=0�=��I:%���`%�=ɺ:=�3�פ�=C�$�9*���?�
>���:%�=�Ö=�Խi� > �9��&=<�͌�kN��+�fz��x�<:o�X�@�*�<�+>.�=@<�~�=R�&����=�6<{�wN��i�=B�j>K5����C>�]='�.=��Ľ��^�M�	=��t���a=�	_�����n_=��=-��=SN���<+���>�-A=��!�Ӊ�=mw��H5�=��=7k:p>מY�Z�m=��=b<ѽ{���R�>gkt���I���<� �󳌾��>xfN��5���<���Y=�����lռײh=��j��[�>�Nl=�t)���<�EY���I�\	F=���;�!=�Z)t>.�F=����q������Aﱽ?j^�����c�a=_K�=h�ލ�=���<�/���(>�D�:�j�=aUL= �O�C͇=Iל��=o�<��h=3�<s�G=��h�=�	$>, ��|l?>j�,�h�=x��=x�2=�UO��!��S��?Ǻ=;�
���D�	�L���=�2]�.���,>��?B;��=�TL��݌=��<=���O<�<0ӽ[v�=���3d����<[�N=���zB�=fļ=��p���ɼw�=��Q�{ox���=�K���.-�r�<@�=�1&<����W����\z>ɟ+�6Ϊ;�[�<x%�=���<�̸=m�N>5�}�F�*��dI�	�=MB���)��(r��,�=�=�=wX�=wId�g{�=�`�q�&��.<����O����)">~��\��=e(��D@'���F���R�@i<�3��yD<P�S���>X����iL>[�0��Ԋ=��-=��>��4Q>�½PN�=�^�=�n���~=j�߽�+��#"����=OT��
����鼽��D�8&�<(��΂�=��,<��=�����=}��=�Դ=�->k���߼[e�=�X�G� � C�*v���X6=���;���=x{�=r�2<��A;�[=C�s>�E��8��6�#��3=Yŋ�J�<V�
>][c�C߬�֥>�4>��'g�={~��{��<�gG��%�� �>�ͅ={�>4��=��\�)V�W*�=�w���u�+�|=ˏ�=�=\=)#!=��> ?"��=È�S|=`�W�Q�C��{>����[�=�J!��OƼxJ>�돽#�>Ό=��=����N�=���<u]�;k��<#��=�qJ>;�A��&=��A</0�=�<?ѧ�d��;+�=��������*[����=^<8�׍�=O�>R/Y<��= u����<3�->6=�'�i�S�>痽o��=������9�i�=��)���M�b/�(�ѻ��;sai���+�qZ��j�F>�0��Ѽ0 ���/(=o�1����Cۜ=����'����8>+��y���.�=B����#��>���/)�=G6!=_[�=��������=�5����5�?�'��	�j��=��=��v��T���t�g$��'�>�#O;�=��5>��<��څֽ��>t�>d���=����W�<�#��~�}��<�x7���;=ss�=�J�<�����㽖���gV"�-5ｳ�<(\�=�h���x<X�	��Ľ���kd����=00>p��pv=৻=R��r�=n�;f1��O[=�	X�a��=ń���>\M1>
"���&�=�J�9<��'������=�hO=Eԋ=��:/�c������!��ж=َ2>� �<Ÿ���&�=��{=�m�fx��k�=���/E��h�=h u���P��L���X�Q�c�[����=�=;�*��.�=������=��=:��=�H�=dY>�.�=#V�<8YO=C_���A��� =�=נ���ט=o��=�'μ�)��;9Q=0>o=���0��w��K��=1*&=�	0���Z<wƜ=�1��騼����z��<��x��+>��E=�� =@����>�n��m�|=ex�o[&�4��=eؼ�l���=���=I_���sg����=գ+�]����ɫ=�[��8��<0�P��
�s�Q��t�<K-]=�8����=�$�<.98�z�=f��=2Q�=<I�����B�>�=�P=�UO=�νյv�Ԉ=��=����@���^>�N���!���=�><(� ���e> 2/=�h����<�j>�W�=B�=��g�ЎټNC�}\f��=>-p\��V�>+�(�#��=B��<�ez��8=+¼/�@�_6�3d��O� =�)e=>3���Wƽz޽� ^�τj=�������,��=��<�O����=�(r=�h��M�>`�+���Mt=%�>C�w==>=2�R=#��á>��9��2P�:���<�MR��G>�>=>�b|=�p��C�B��<Ge�i<8��=��>�S=�O�=J$��u��-�u������̘=z7��ּ�Vr�ҍ��]��=۞�:F1=���=�������k�>�j>i� �G�`7E>O"�����{jR�uB>�T^<BXZ�@�M�����~޼[׽b��<��=�*�=��<K݃�r�F*��b�� ��/>�՟�C���_�c�6=c*�=࿈���e��>Zs�=?�<?'>;���"�t��Y>s}$<�ٙ���<0����_>Yd�=����'�<�i��t�S�;�=#���#�4<9p�=��j��r��(*�0 <�э=�v<��g>�
�<���I�=���?�wP��M���=����O==o�o=��A�;X�=@�\� �߼>t">?�M>CE+>R~e=O�6<h�:=ˇ������9S=pBm=5B<=q�����=���=��8=8���=<���=׹���9%�_,A>Px�Ň
>��s���jRh�:��=�����=d/=~=��b�R\�<�l�=k�I�W�k=��<��*w�=��Խ��?�;�6=�q+�1ǋ=!2�F��;ϣ
��cf��ߦ=�k�=�:�=>���7�ͩ�=3=�����>Z�����WJ�/Mҽ/ձ����=��>�(�<��<�E�����oU��82�=94�s$���3�hՎ����D�(=M�=L���xS=�"=�g�<���=�������=�f*�����U{
���X�l���kR1;ݶ;=�B���7>a/��[��=�[������!��>��j��ӏ��h=�w�<m���pz�����O
�<ͩ佼����o��
��=����?�������3=��]����=�½�+'>2���'K>��<���5�N���� >�E;7Ռ<2��_rO������� >�����>��l=�8ջR�����=T���޽��d�a�K;O�>��u�����ጽ�!c>�@�<��u�����T+ �e]<�A�=חd=>��]1>߷=�Y}9�Bk�rtU�O��j'>�Zt=G��ߓ��t̽����/�Y���=��0�}?������<<^$�w2�Oc);*_�=Կ>W2.�A$@��������=�e�=Kn�jx���$=� ��L=�ʽe/��bq�������=���@g�=�m��#~�=ao>�u�SU�=���=�,x= �ͽO͑=]�<^Y�ޡA>:K�=%GM���u �{�s=K�*>@��=m�=���=5�o����=W��� >dI�/�=�j�>�+/:rp���B�����J�=���=�䗼kH'=N�������M<���=g�0�=�ӽ�Yɻ�B��*
_output_shapes

:!C*
dtype0
~

Variable_4
VariableV2*
shape
:!C*
dtype0*
_output_shapes

:!C*
shared_name *
	container 
�
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
T0*
_class
loc:@Variable_4*
_output_shapes

:!C*
use_locking(*
validate_shape(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:!C
�
Variable_5/initial_valueConst*�
value�B�C"��#>͵��=��=J��=~m��Fv�����˫0<#x�<�tB>0�M;�]<̝��g6<��)>��=/� ����<='C�~��=7���n�=T����t��*x=���� />�%����=4�*>�"�=p�q>�B>#�w�Tw<����>	5�?�=BP�Oꍾ	�8=�%z���>L�>�ߘ=�*>�e��m�];�3>�x��;�_ǽ�ٽ%M>����;2b�<���=�@�=~�=T�9=�\��e�=真�BQ��>?%=K,>*
_output_shapes

:C*
dtype0
~

Variable_5
VariableV2*
_output_shapes

:C*
dtype0*
shared_name *
	container *
shape
:C
�
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
validate_shape(*
T0*
_output_shapes

:C*
use_locking(*
_class
loc:@Variable_5
o
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes

:C
�
Variable_6/initial_valueConst*�
value�B�C"�����U�R=	<�<�|��K�=�]Ͻ����'舽d�:�z?;J��=��߽7)���M>?�,=��](V�;�_=B�1>����k�n<O��<(�����:v⩽��m=�J�P�Խ��=�̼z�0|5��[=K������:�M>e���2�x>��E��ݵ=�I�=C�X=�c�=+��=�ۋ<��=�սg.�<��f=�1�p��=�
�;W�&�7�_�:�(���T<t
��r>���=�8<���T��\d>�y��%C��5����<�4��*
_output_shapes

:C*
dtype0
~

Variable_6
VariableV2*
shared_name *
shape
:C*
dtype0*
_output_shapes

:C*
	container 
�
Variable_6/AssignAssign
Variable_6Variable_6/initial_value*
T0*
validate_shape(*
_class
loc:@Variable_6*
_output_shapes

:C*
use_locking(
o
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes

:C
m
Variable_7/initial_valueConst*
valueB*?��*
_output_shapes

:*
dtype0
~

Variable_7
VariableV2*
	container *
dtype0*
_output_shapes

:*
shape
:*
shared_name 
�
Variable_7/AssignAssign
Variable_7Variable_7/initial_value*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
o
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes

:*
T0
�
MatMulMatMulPlaceholderVariable_4/read*'
_output_shapes
:���������C*
T0*
transpose_a( *
transpose_b( 
U
addAddMatMulVariable_5/read*'
_output_shapes
:���������C*
T0
C
ReluReluadd*
T0*'
_output_shapes
:���������C
�
MatMul_1MatMulReluVariable_6/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
add_1AddMatMul_1Variable_7/read*'
_output_shapes
:���������*
T0
G
Relu_1Reluadd_1*'
_output_shapes
:���������*
T0
�
$mean_squared_error/SquaredDifferenceSquaredDifferenceRelu_1Placeholder_1*
T0*'
_output_shapes
:���������
t
/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
u
3mean_squared_error/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
�
mean_squared_error/Cast/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/Cast/x*
T0*'
_output_shapes
:���������
�
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
�
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
�
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$mean_squared_error/num_present/EqualEqualmean_squared_error/Cast/x&mean_squared_error/num_present/Equal/y*
_output_shapes
: *
T0
�
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
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
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
_output_shapes
: *

index_type0*
T0
�
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
�
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
�
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
T0*
_output_shapes
:
�
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*

index_type0*
T0*'
_output_shapes
:���������
�
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
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
mean_squared_error/valueDivNoNanmean_squared_error/Sum_1mean_squared_error/num_present*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
�
=gradients/mean_squared_error/value_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/mean_squared_error/value_grad/Shape/gradients/mean_squared_error/value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/mean_squared_error/value_grad/div_no_nanDivNoNangradients/Fillmean_squared_error/num_present*
T0*
_output_shapes
: 
�
+gradients/mean_squared_error/value_grad/SumSum2gradients/mean_squared_error/value_grad/div_no_nan=gradients/mean_squared_error/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
�
/gradients/mean_squared_error/value_grad/ReshapeReshape+gradients/mean_squared_error/value_grad/Sum-gradients/mean_squared_error/value_grad/Shape*
_output_shapes
: *
T0*
Tshape0
m
+gradients/mean_squared_error/value_grad/NegNegmean_squared_error/Sum_1*
_output_shapes
: *
T0
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
-gradients/mean_squared_error/value_grad/Sum_1Sum+gradients/mean_squared_error/value_grad/mul?gradients/mean_squared_error/value_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
�
1gradients/mean_squared_error/value_grad/Reshape_1Reshape-gradients/mean_squared_error/value_grad/Sum_1/gradients/mean_squared_error/value_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/value_grad/Reshape2^gradients/mean_squared_error/value_grad/Reshape_1
�
@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/value_grad/Reshape9^gradients/mean_squared_error/value_grad/tuple/group_deps*B
_class8
64loc:@gradients/mean_squared_error/value_grad/Reshape*
T0*
_output_shapes
: 
�
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/value_grad/Reshape_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*
_output_shapes
: *D
_class:
86loc:@gradients/mean_squared_error/value_grad/Reshape_1
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients/mean_squared_error/value_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
dtype0*
valueB *
_output_shapes
: 
�
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
�
3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
�
+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
_output_shapes
:*
T0*
out_type0
�
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
�
+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/Cast/x*
T0*'
_output_shapes
:���������
�
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*'
_output_shapes
:���������*
T0
�
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
�
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
T0*
_output_shapes
: 

9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
�
;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
�
Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
7gradients/mean_squared_error/SquaredDifference_grad/MulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
7gradients/mean_squared_error/SquaredDifference_grad/subSubRelu_1Placeholder_1?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/Mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
�
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*'
_output_shapes
:���������*
Tshape0
�
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
�
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*
T0*'
_output_shapes
:���������
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
valueB"      *
_output_shapes
:*
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes

:*
Tshape0*
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
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*
_output_shapes

:*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_6/read*'
_output_shapes
:���������C*
T0*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:C*
transpose_b( *
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������C
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes

:C
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:���������C
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
k
gradients/add_grad/Shape_1Const*
dtype0*
valueB"   C   *
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:���������C*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes

:C*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������C*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes

:C*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *'
_output_shapes
:���������!*
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes

:!C*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������!
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:!C*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
b
GradientDescent/learning_rateConst*
valueB
 *
�#<*
_output_shapes
: *
dtype0
�
6GradientDescent/update_Variable_4/ApplyGradientDescentApplyGradientDescent
Variable_4GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
_output_shapes

:!C
�
6GradientDescent/update_Variable_5/ApplyGradientDescentApplyGradientDescent
Variable_5GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_5*
T0*
_output_shapes

:C
�
6GradientDescent/update_Variable_6/ApplyGradientDescentApplyGradientDescent
Variable_6GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
_output_shapes

:C*
use_locking( *
T0
�
6GradientDescent/update_Variable_7/ApplyGradientDescentApplyGradientDescent
Variable_7GradientDescent/learning_rate/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
_class
loc:@Variable_7*
T0
�
GradientDescentNoOp7^GradientDescent/update_Variable_4/ApplyGradientDescent7^GradientDescent/update_Variable_5/ApplyGradientDescent7^GradientDescent/update_Variable_6/ApplyGradientDescent7^GradientDescent/update_Variable_7/ApplyGradientDescent
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_cbc99fd60b25439499653213e60d8d39/part*
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
: *
value	B :*
dtype0
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
save/SaveV2/tensor_namesConst"/device:CPU:0*
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
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*#
valueBB B B B B B B B 
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
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*
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
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*#
valueBB B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*
_output_shapes

:!C
�
save/Assign_1Assign
Variable_1save/RestoreV2:1*
_output_shapes

:C*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
�
save/Assign_2Assign
Variable_2save/RestoreV2:2*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2*
_output_shapes

:C
�
save/Assign_3Assign
Variable_3save/RestoreV2:3*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3*
_output_shapes

:
�
save/Assign_4Assign
Variable_4save/RestoreV2:4*
_output_shapes

:!C*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
�
save/Assign_5Assign
Variable_5save/RestoreV2:5*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes

:C*
use_locking(*
T0
�
save/Assign_6Assign
Variable_6save/RestoreV2:6*
_output_shapes

:C*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
�
save/Assign_7Assign
Variable_7save/RestoreV2:7*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"
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

mean_squared_error/value:0*�
serving_defaultv
-
input$
Placeholder:0���������!)
output
Relu_1:0���������tensorflow/serving/predict