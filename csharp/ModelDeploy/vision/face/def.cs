namespace ModelDeploy.vision.face;

public static class FaceConstants
{
    public const int MD_FACE_DETECT = 0b00000001;
    public const int MD_FACE_LANDMARK = 0b00000001 << 1;
    public const int MD_FACE_RECOGNITION = 0b00000001 << 2;
    public const int MD_FACE_ANTI_SPOOFING = 0b00000001 << 3;
    public const int MD_FACE_QUALITY_EVALUATE = 0b00000001 << 4;
    public const int MD_FACE_AGE_ATTRIBUTE = 0b00000001 << 5;
    public const int MD_FACE_GENDER_ATTRIBUTE = 0b00000001 << 6;
    public const int MD_FACE_EYE_STATE = 0b00000001 << 7;

    public const int MD_MASK = MD_FACE_DETECT | MD_FACE_LANDMARK | MD_FACE_RECOGNITION |
                               MD_FACE_ANTI_SPOOFING | MD_FACE_QUALITY_EVALUATE |
                               MD_FACE_AGE_ATTRIBUTE | MD_FACE_GENDER_ATTRIBUTE |
                               MD_FACE_EYE_STATE;
}